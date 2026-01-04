"""诗词理解与推理评测 Few-shot Baseline 脚本。

本脚本会请求一个 OpenAI 兼容的聊天接口（通常是本地 vLLM 服务），生成以下内容：
- `qa_words` 的词语解释（ans_qa_words）
- `qa_sents` 的句子白话翻译（ans_qa_sents）
- 情感选项（choose_id，A/B/C/D）

Few-shot 部分会从训练集（`data/train-data/**/train.json`）中检索 K 条相似样本。
检索方法为轻量级的 token 重叠 + IDF 打分，然后把这些示例的关键字段以紧凑形式拼进提示词。

运行方式（示例）：
    1）启动 vLLM 的 OpenAI 兼容服务：
       `vllm serve /path/to/Qwen2.5-7B-Instruct --served-model-name qwen2.5-7b --max_model_len 20000`
    2）运行本脚本生成 `submit.json`。

重要说明：
- 文件名包含连字符（`few-shot.py`），按脚本方式执行即可，不建议作为模块被 import。
"""

# 使用 vllm,使用方式见GitHub vllm
# 这是我的服务器启动
# vllm serve /mnt/home/user04/CCL/model/Qwen2.5-7B-Instruct  --served-model-name qwen2.5-7b   --max_model_len 20000
import os
import json
from openai import OpenAI
from tqdm import tqdm
import re
import math
from collections import Counter

# 设置 HTTP 代理
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key
)
models = client.models.list()
model = models.data[0].id
print(model)

# few-shot 配置：从训练集检索 K 条相似示例（懒加载，避免 import 时未定义函数）
FEWSHOT_K = 5
TRAIN_EXAMPLES = None
TRAIN_DOC_TOKENS = None
TRAIN_IDF = None


def _tokenize(text: str):
    """为检索任务对中文文本做简单分词。

    策略：
        - 去除空白符；
        - 保留中文字符与字母数字；
        - 同时输出单字（unigram）与相邻双字（bigram）。

    参数：
        text：输入文本。

    返回：
        list[str]：token 列表。
    """
    # 简单中文 token：单字 + 相邻双字（更利于相似检索）
    text = re.sub(r"\s+", "", str(text))
    chars = [c for c in text if ("\u4e00" <= c <= "\u9fff") or c.isalnum()]
    tokens = []
    tokens.extend(chars)
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])
    return tokens


def _load_train_examples(train_root: str = "data/train-data"):
    """递归扫描训练目录并加载训练样本。

    仓库中训练数据按类别存放在 `data/train-data/` 下，每个子目录包含一个 `train.json`。
    本函数会递归查找所有名为 `train.json` 的文件并汇总样本。

    参数：
        train_root：训练数据根目录。

    返回：
        list[dict]：训练样本列表。
    """
    examples = []
    for root, _, files in os.walk(train_root):
        for name in files:
            if name.lower() != "train.json":
                continue
            fp = os.path.join(root, name)
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                examples.extend([x for x in data if isinstance(x, dict)])
            elif isinstance(data, dict):
                examples.append(data)
    return examples


def _build_retriever(train_examples):
    """构建轻量级的 IDF 检索器索引。

    对每条训练样本，基于 (title, author, content) 构造 token 集合。
    对查询文本的打分为：与文档 token 的重叠部分的 IDF 权重之和。

    参数：
        train_examples（list[dict]）：训练样本。

    返回：
        tuple[list[set[str]], dict[str, float]]：
            - doc_tokens：每条训练样本对应的 token 集合
            - idf：token -> idf 权重
    """
    # 轻量检索：idf(重叠 token) 求和
    doc_tokens = []
    df = Counter()
    for ex in train_examples:
        text = "\n".join(
            [
                str(ex.get("title", "")),
                str(ex.get("author", "")),
                str(ex.get("content", "")),
            ]
        )
        toks = set(_tokenize(text))
        doc_tokens.append(toks)
        df.update(toks)
    n = max(len(train_examples), 1)
    idf = {t: (math.log((n + 1) / (c + 1)) + 1.0) for t, c in df.items()}
    return doc_tokens, idf


def _ensure_fewshot_loaded():
    """按需（懒加载）初始化并缓存训练样本与检索索引。"""
    global TRAIN_EXAMPLES, TRAIN_DOC_TOKENS, TRAIN_IDF
    if TRAIN_EXAMPLES is not None:
        return
    TRAIN_EXAMPLES = _load_train_examples("data/train-data")
    TRAIN_DOC_TOKENS, TRAIN_IDF = _build_retriever(TRAIN_EXAMPLES)
    print(f"Loaded train examples: {len(TRAIN_EXAMPLES)}")


def _topk_examples(query_text: str, train_examples, doc_tokens, idf, k: int = 3):
    """返回用于 few-shot 提示的 top-k 相似训练样本。"""
    q = set(_tokenize(query_text))
    scored = []
    for i, dt in enumerate(doc_tokens):
        overlap = q & dt
        score = 0.0
        for t in overlap:
            score += idf.get(t, 0.0)
        scored.append((score, i))
    scored.sort(reverse=True)
    out = []
    for score, i in scored[:k]:
        if score <= 0:
            continue
        out.append(train_examples[i])
    return out


def _format_fewshot_block(examples):
    """把检索到的示例格式化为紧凑的 few-shot 提示块。"""
    if not examples:
        return ""
    lines = []
    lines.append("【参考示例（来自训练数据，用于学习释义/译文/情感风格）】")
    for i, ex in enumerate(examples, 1):
        # 只放关键字段，避免上下文过长
        lines.append(f"示例{i}：")
        lines.append(f"title：{ex.get('title', '')}")
        lines.append(f"author：{ex.get('author', '')}")
        lines.append(f"content：{ex.get('content', '')}")
        kw = ex.get("keywords", {})
        if isinstance(kw, dict) and kw:
            # 只取前若干条
            items = list(kw.items())[:8]
            kw_preview = {k: v for k, v in items}
            lines.append(f"keywords：{json.dumps(kw_preview, ensure_ascii=False)}")
        tr = ex.get("trans", "")
        if tr:
            lines.append(f"trans：{tr}")
        emo = ex.get("emotion", "")
        if emo:
            lines.append(f"emotion：{emo}")
        lines.append("---")
    return "\n".join(lines) + "\n"


def get_response(data):
    """为单条评测样本生成答案。

    参数：
        data (dict)：评测样本。

    返回：
        dict：符合提交要求的字典，包含：
            - idx
            - ans_qa_words
            - ans_qa_sents
            - choose_id

    说明：
        - 通过提示词强约束模型只输出 JSON；
        - 内置重试机制，并从输出中提取第一个 JSON 对象。
    """
    # few-shot：从训练集中检索相似示例，作为风格参考
    _ensure_fewshot_loaded()
    query_text = "\n".join(
        [
            str(data.get("title", "")),
            str(data.get("author", "")),
            str(data.get("content", "")),
            " ".join(map(str, data.get("qa_words", []) or [])),
            " ".join(map(str, data.get("qa_sents", []) or [])),
        ]
    )
    fewshot = _topk_examples(query_text, TRAIN_EXAMPLES, TRAIN_DOC_TOKENS, TRAIN_IDF, k=FEWSHOT_K)
    fewshot_block = _format_fewshot_block(fewshot)

    prompt = f"""
你是一个古诗词专家，现在有一些古诗词需要你的帮助。

我会给你提供一个 JSON 数据，字段可能包括：
- "idx" 或 "index"：古诗词的序号（两者可能出现其一）
- "title"：古诗词的标题
- "author"：古诗词的作者
- "content"：古诗词的内容
- "qa_words"：需要你解释含义的词语列表
- "qa_sents"：需要你给出白话译文的句子列表
- "choose"：情感选项（通常为 A/B/C/D 的字典）

{fewshot_block}

【当前需要作答的评测样本】
这是我的数据：
{data}

【你的任务】
请你根据提供的数据，生成如下 JSON 格式的结果：
1) "ans_qa_words"：对 qa_words 词语的含义进行解释（结合本诗语境）
2) "ans_qa_sents"：对 qa_sents 句子提供白话文译文（逐句翻译）
3) "choose_id"：从 choose 选项中选择最符合该古诗词情感的标号，仅输出对应字母（A/B/C/D）

【写作要求（更贴近评测）】
A. 词语释义（ans_qa_words）
- 释义要像“词典释义”，短、准、明确；不要写赏析、不要写背景故事。
- 优先给出在本诗语境下的具体含义，可用“这里指……”但不要太长。
- 每个词语建议 6~24 个汉字；避免冗长解释。

B. 句子翻译（ans_qa_sents）
- 逐句直译为主，简洁通顺；不要扩写成一大段赏析。
- 不要引入原句没有的新意象/新情节；地名、人名、典故名可保留或简要解释。
- 翻译尽量贴近训练集译文风格：清晰叙述、不堆砌形容词。

C. 情感选择（choose_id）
- 先整体判断全诗主情感，再在 choose 中选择最符合的“字母键”。
- 若多项接近，选择最主要、最贯穿全诗的情绪选项。

【必须严格遵守的输出规则】
1) 仅返回 JSON 结果，不需要任何额外解释、标题、前后缀文本。
2) 不要输出 Markdown，不要输出 ```。
3) 输出必须是合法 JSON：使用双引号、无尾逗号、无注释。
4) "ans_qa_words" 必须包含 qa_words 的全部词语作为 key（一个都不能少）；"ans_qa_sents" 必须包含 qa_sents 的全部原句作为 key（一个都不能少）。
     - 如果某个词/句你不确定，也必须输出一个尽量合理的答案（不要省略 key）。
5) "choose_id" 只能输出 "A"/"B"/"C"/"D" 之一，不要输出选项文本。
6) 必须用中文作答。

【JSON 输出格式示例（照此结构输出）】
{{
    "idx": 0,
    "ans_qa_words": {{
        "词语1": "词语1的含义",
        "词语2": "词语2的含义"
    }},
    "ans_qa_sents": {{
        "句子1": "句子1的白话文翻译",
        "句子2": "句子2的白话文翻译"
    }},
    "choose_id": "A"
}}

现在开始作答：只输出 JSON。
    """
    for _ in range(3):  # 三次重传机制
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            content = response.choices[0].message.content.strip()
            # 使用正则提取 JSON 格式的内容（匹配 { } 之间的内容）
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                answer = match.group(0)  # 获取匹配的 JSON 字符串
                answer = answer.strip()  # 去除前后空格
                answer = json.loads(answer)
                if isinstance(answer, dict) and "idx" not in answer:
                    answer["idx"] = data.get("idx", data.get("index", ""))
                return answer
        except json.JSONDecodeError:
            continue
    return {
        "idx": data.get("idx", data.get("index", "")),
        "ans_qa_words": {},
        "ans_qa_sents": {},
        "choose_id": -1
    }


def main():
    """程序入口：读取评测数据，逐条处理并写出 `submit.json`。"""
    # 读取输入数据
    output_path = "submit.json"
    input_path = "data/eval_data.json"
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)  # [:10] 可以打印几条数据测试
    output_data = []
    for data in tqdm(input_data, desc="Processing"):
        answer = get_response(data)
        print(answer)
        if answer:
            output_data.append(answer)
        else:
            print(f"未能生成答案的数据")

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"处理完成。共处理 {len(output_data)} 条数据，输出已保存到 {output_path}。")


if __name__ == "__main__":
    main()
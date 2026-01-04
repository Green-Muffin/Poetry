"""诗词理解与推理评测主方案脚本。

该脚本实现了比 `baseline.py` 更强的基线，主要组合了：
- 轻量级检索（token 重叠 + IDF）覆盖 `data/train-data/**/train.json` 下的全部训练切分；
- 从训练集中挖掘全局的“词语 -> 释义”字典，用于高置信度的词语释义；
- 将任务拆成三段 LLM 调用，降低子任务之间的相互干扰：
    1）词语释义（ans_qa_words）
    2）句子翻译（ans_qa_sents）
    3）情感选择（映射到 A/B/C/D 的 choose_id）

推理服务需要是 OpenAI 兼容接口（例如本地 vLLM 服务）。代码通过重试与 JSON
抽取来提高对异常返回的鲁棒性。

输出：
    按要求写出 `submit.json`。
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
TRAIN_KEYWORD_DICT = None

# 词典释义仅在高置信时直接使用（避免一词多义带来的错释义拉低 BLEU）
WORD_DICT_MIN_RATIO = 0.70
WORD_DICT_MIN_COUNT = 3

# 翻译阶段仅在检索足够相似时才给 trans few-shot（降低带偏风险）
TRANS_FEWSHOT_MIN_SCORE = 1.20  # 归一化后的 overlap-idf 分数阈值


def _extract_json(text: str):
    """从模型输出中提取第一个 JSON 对象。

    模型有时会把 JSON 包在 Markdown 代码块中，或在前后附加一些文本。
    本函数会先去掉常见的代码块包裹，再用正则匹配第一个 `{...}` 并尝试解析。

    参数：
        text：模型原始输出文本。

    返回：
        dict | None：解析成功返回字典；否则返回 None。
    """
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_llm(prompt: str, max_retries: int = 3):
    """调用聊天接口并带重试。

    参数：
        prompt：用户提示词。
        max_retries：请求失败时的重试次数。

    返回：
        str：模型返回的原始文本（可能为空）。

    说明：
        - 为了批量推理不中断，这里刻意捕获了较宽泛的异常；
        - temperature 设为 0.0 以获得更稳定的输出。
    """
    last_text = ""
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.0,
                max_tokens=800,
            )
            last_text = response.choices[0].message.content.strip()
            return last_text
        except Exception:
            continue
    return last_text


def _tokenize(text: str):
    """为检索任务对中文文本做简单分词（单字 + 相邻双字）。"""
    # 简单中文 token：单字 + 相邻双字（更利于相似检索）
    text = re.sub(r"\s+", "", str(text))
    chars = [c for c in text if ("\u4e00" <= c <= "\u9fff") or c.isalnum()]
    tokens = []
    tokens.extend(chars)
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])
    return tokens


def _load_train_examples(train_root: str = "data/train-data"):
    """遍历 `train_root` 并加载其中所有 `train.json`。

    返回：
        list[dict]：汇总后的训练样本列表。
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
    """构建检索索引（每条样本的 token 集合 + 全局 IDF 权重）。"""
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
    """按需（懒加载）初始化检索索引与全局词语释义字典。

    副作用：
        会填充全局变量 TRAIN_EXAMPLES / TRAIN_DOC_TOKENS / TRAIN_IDF / TRAIN_KEYWORD_DICT。
    """
    global TRAIN_EXAMPLES, TRAIN_DOC_TOKENS, TRAIN_IDF, TRAIN_KEYWORD_DICT
    if TRAIN_EXAMPLES is not None and TRAIN_KEYWORD_DICT is not None:
        return
    TRAIN_EXAMPLES = _load_train_examples("data/train-data")
    TRAIN_DOC_TOKENS, TRAIN_IDF = _build_retriever(TRAIN_EXAMPLES)
    # 构建“词语->释义”全局词典：存 top1/total 以便做置信过滤
    keyword_def_counts = {}
    keyword_total = Counter()
    for ex in TRAIN_EXAMPLES:
        kw = ex.get("keywords", {})
        if not isinstance(kw, dict):
            continue
        for w, d in kw.items():
            w = str(w).strip()
            d = str(d).strip()
            if not w or not d:
                continue
            keyword_def_counts.setdefault(w, Counter())[d] += 1
            keyword_total[w] += 1
    TRAIN_KEYWORD_DICT = {}
    for w, cnt in keyword_def_counts.items():
        top_def, top_cnt = cnt.most_common(1)[0]
        TRAIN_KEYWORD_DICT[w] = {
            "def": top_def,
            "top": int(top_cnt),
            "total": int(keyword_total.get(w, 0)),
        }
    print(f"Loaded train examples: {len(TRAIN_EXAMPLES)}")


def _topk_examples(query_text: str, train_examples, doc_tokens, idf, k: int = 3):
    """返回 top-k 相似样本（未归一化的 overlap-idf 分数）。"""
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


def _topk_examples_scored(query_text: str, train_examples, doc_tokens, idf, k: int = 3):
    """返回 top-k 相似样本，并给出归一化分数。

    归一化方式：用 raw overlap-idf 除以查询 token 数，以避免过长输入占优。

    参数：
        query_text：查询文本。
        train_examples：训练样本。
        doc_tokens：训练样本的 token 集合。
        idf：token -> idf 权重。
        k：Top-K。

    返回：
        list[tuple[float, dict]]：(score, example) 列表。
    """
    q = set(_tokenize(query_text))
    denom = max(len(q), 1)
    scored = []
    for i, dt in enumerate(doc_tokens):
        overlap = q & dt
        raw = 0.0
        for t in overlap:
            raw += idf.get(t, 0.0)
        score = raw / denom
        scored.append((score, i))
    scored.sort(reverse=True)
    out = []
    for score, i in scored[:k]:
        if score <= 0:
            continue
        out.append((score, train_examples[i]))
    return out


def _format_fewshot_block(examples, mode: str):
    """把 few-shot 示例格式化为提示词块。

    参数：
        examples：检索到的示例。
        mode：'words' 或 'sents'。不同任务只保留不同字段以降低噪声。

    返回：
        str：可插入到提示词中的文本块。

    说明：
        few-shot 很容易把输出带偏，因此这里会按任务选择性地放字段。
    """
    if not examples:
        return ""
    lines = []
    if mode == "words":
        lines.append("【参考示例（仅用于学习 keywords 释义风格）】")
        for i, ex in enumerate(examples, 1):
            lines.append(f"示例{i}：")
            lines.append(f"content：{ex.get('content', '')}")
            kw = ex.get("keywords", {})
            if isinstance(kw, dict) and kw:
                items = list(kw.items())[:10]
                kw_preview = {k: v for k, v in items}
                lines.append(f"keywords：{json.dumps(kw_preview, ensure_ascii=False)}")
            lines.append("---")
    else:
        # sents：只给 1 个风格示例（避免译文被带偏）
        ex = examples[0]
        lines.append("【参考示例（仅用于学习译文风格）】")
        lines.append(f"content：{ex.get('content', '')}")
        tr = ex.get("trans", "")
        if tr:
            lines.append(f"trans：{tr}")
        lines.append("---")
    return "\n".join(lines) + "\n"


def get_response(data):
    """为单条评测样本生成完整提交结果（词语/句子/情感）。

    该函数刻意做了较强的“兜底/防御式”处理：
    - 优先使用高置信度的训练集词典释义；
    - 仅对缺失项调用模型；
    - 对空输出进行二次补全；
    - 保证所有必需字段与 key 都存在且为非空。

    参数：
        data (dict)：评测样本。

    返回：
        dict：单条提交结果。
    """
    # few-shot：从训练集中检索相似示例，作为风格参考
    _ensure_fewshot_loaded()

    idx = data.get("idx", data.get("index", ""))
    query_text = "\n".join(
        [
            str(data.get("title", "")),
            str(data.get("author", "")),
            str(data.get("content", "")),
            " ".join(map(str, data.get("qa_words", []) or [])),
            " ".join(map(str, data.get("qa_sents", []) or [])),
        ]
    )
    fewshot_scored = _topk_examples_scored(query_text, TRAIN_EXAMPLES, TRAIN_DOC_TOKENS, TRAIN_IDF, k=FEWSHOT_K)
    fewshot = [ex for _, ex in fewshot_scored]
    top_score = fewshot_scored[0][0] if fewshot_scored else 0.0

    # 词语释义优先走训练集词典，缺的再调用模型
    qa_words = data.get("qa_words", []) or []
    ans_words = {}
    for w in qa_words:
        w_str = str(w)
        info = TRAIN_KEYWORD_DICT.get(w_str)
        if isinstance(info, dict):
            top = int(info.get("top", 0))
            total = int(info.get("total", 0))
            ratio = (top / total) if total else 0.0
            if top >= WORD_DICT_MIN_COUNT and ratio >= WORD_DICT_MIN_RATIO:
                ans_words[w_str] = str(info.get("def", "")).strip()
            else:
                ans_words[w_str] = ""
        else:
            ans_words[w_str] = ""

    missing_words = [w for w in qa_words if not ans_words.get(str(w))]
    if missing_words:
        fewshot_block_words = _format_fewshot_block(fewshot, mode="words")
        prompt_words = f"""
你是古诗词词语释义助手。

【诗词】
title：{data.get('title', '')}
author：{data.get('author', '')}
content：{data.get('content', '')}

{fewshot_block_words}

【需要解释的词语】
{json.dumps(missing_words, ensure_ascii=False)}

【任务】
请仅对以上词语给出“在本诗语境下”的中文释义，尽量短、准，像词典释义。

【输出要求】
1) 只输出严格 JSON。
2) 必须包含且只包含："idx" 和 "ans_qa_words"。
3) ans_qa_words 必须覆盖所有输入词语作为 key（一个都不能少）。

输出示例：{{"idx": {idx}, "ans_qa_words": {{"词语": "释义"}}}}

现在开始作答：只输出 JSON。
"""
        words_text = _call_llm(prompt_words, max_retries=3)
        words_json = _extract_json(words_text) or {}
        if isinstance(words_json, dict) and isinstance(words_json.get("ans_qa_words", None), dict):
            for k, v in words_json["ans_qa_words"].items():
                k = str(k)
                ans_words[k] = str(v).strip()

    # 二次补全：如果仍有空释义，强制再问一次（避免评测判定“无回答”）
    empty_words = [str(w) for w in qa_words if not str(ans_words.get(str(w), "")).strip()]
    if empty_words:
        prompt_words2 = f"""
你是古诗词词语释义助手。

【诗词】
content：{data.get('content', '')}

【需要解释的词语】
{json.dumps(empty_words, ensure_ascii=False)}

【任务】
必须为每个词语给出非空中文释义（结合本诗语境），每条尽量 6~24 个汉字，像词典释义。

【输出要求】
只输出严格 JSON，格式：{{"idx": {idx}, "ans_qa_words": {{"词语": "释义"}}}}
"""
        words_text2 = _call_llm(prompt_words2, max_retries=2)
        words_json2 = _extract_json(words_text2) or {}
        if isinstance(words_json2, dict) and isinstance(words_json2.get("ans_qa_words", None), dict):
            for k, v in words_json2["ans_qa_words"].items():
                k = str(k)
                vv = str(v).strip()
                if vv:
                    ans_words[k] = vv

    # 补齐 key（避免漏字段被评测扣分）
    for w in qa_words:
        if str(w) not in ans_words:
            ans_words[str(w)] = ""

    # 最终兜底：确保每个词语释义非空
    for w in qa_words:
        ww = str(w)
        if not str(ans_words.get(ww, "")).strip():
            ans_words[ww] = f"这里指{ww}的意思"

    # 句子翻译单独一调用（减少与词语释义互相干扰）
    qa_sents = data.get("qa_sents", []) or []
    # 仅当检索足够相似时才给 trans few-shot（否则不加，避免带偏）
    fewshot_block_sents = ""
    if top_score >= TRANS_FEWSHOT_MIN_SCORE and fewshot:
        fewshot_block_sents = _format_fewshot_block(fewshot[:1], mode="sents")

    prompt_sents = f"""
你是一个古诗词专家，现在有一些古诗词需要你的帮助。

我会给你提供一个 JSON 数据，字段可能包括：
- "idx" 或 "index"：古诗词的序号（两者可能出现其一）
- "title"：古诗词的标题
- "author"：古诗词的作者
- "content"：古诗词的内容
- "qa_words"：需要你解释含义的词语列表
- "qa_sents"：需要你给出白话译文的句子列表
- "choose"：情感选项（通常为 A/B/C/D 的字典）

{fewshot_block_sents}

【当前需要作答的评测样本】
这是我的数据：
{data}

【你的任务】
请你根据提供的数据，生成如下 JSON 格式的结果（本次只做句子翻译）：
1) "ans_qa_sents"：对 qa_sents 句子提供白话文译文（逐句翻译）

【写作要求（更贴近评测）】
A. 词语释义（ans_qa_words）
- 本次不需要输出 ans_qa_words。

B. 句子翻译（ans_qa_sents）
- 逐句直译为主，简洁通顺；不要扩写成一大段赏析。
- 不要引入原句没有的新意象/新情节；地名、人名、典故名可保留或简要解释。
- 翻译尽量贴近训练集译文风格：清晰叙述、不堆砌形容词。

C. 情感选择（choose_id）
- 本次不需要输出 choose_id。

【必须严格遵守的输出规则】
1) 仅返回 JSON 结果，不需要任何额外解释、标题、前后缀文本。
2) 不要输出 Markdown，不要输出 ```。
3) 输出必须是合法 JSON：使用双引号、无尾逗号、无注释。
4) "ans_qa_words" 必须包含 qa_words 的全部词语作为 key（一个都不能少）；"ans_qa_sents" 必须包含 qa_sents 的全部原句作为 key（一个都不能少）。
     - 如果某个词/句你不确定，也必须输出一个尽量合理的答案（不要省略 key）。
5) 本次不要输出 "ans_qa_words" 和 "choose_id" 字段。
6) 必须用中文作答。

【JSON 输出格式示例（照此结构输出）】
{{
    "idx": 0,
    "ans_qa_sents": {{
        "句子1": "句子1的白话文翻译",
        "句子2": "句子2的白话文翻译"
    }}
}}

现在开始作答：只输出 JSON。
    """

    sents_text = _call_llm(prompt_sents, max_retries=3)
    sents_json = _extract_json(sents_text) or {}
    if not isinstance(sents_json, dict):
        sents_json = {}
    ans_sents = sents_json.get("ans_qa_sents", {}) if isinstance(sents_json.get("ans_qa_sents", {}), dict) else {}

    for s in qa_sents:
        if str(s) not in ans_sents:
            ans_sents[str(s)] = ""

    # 二次补全：如果仍有空译文，强制再问一次（避免评测判定“无回答”）
    empty_sents = [str(s) for s in qa_sents if not str(ans_sents.get(str(s), "")).strip()]
    if empty_sents:
        prompt_sents2 = f"""
你是古诗词白话翻译助手。

【诗词】
content：{data.get('content', '')}

【需要翻译的句子】
{json.dumps(empty_sents, ensure_ascii=False)}

【任务】
必须逐句给出非空白话译文，直译为主，简洁通顺，不要赏析，不要扩写。

【输出要求】
只输出严格 JSON，格式：{{"idx": {idx}, "ans_qa_sents": {{"原句": "译文"}}}}
"""
        sents_text2 = _call_llm(prompt_sents2, max_retries=2)
        sents_json2 = _extract_json(sents_text2) or {}
        if isinstance(sents_json2, dict) and isinstance(sents_json2.get("ans_qa_sents", None), dict):
            for k, v in sents_json2["ans_qa_sents"].items():
                k = str(k)
                vv = str(v).strip()
                if vv:
                    ans_sents[k] = vv

    # 最终兜底：确保每个句子译文非空（最差也返回原句，防止评测报“无回答”）
    for s in qa_sents:
        ss = str(s)
        if not str(ans_sents.get(ss, "")).strip():
            ans_sents[ss] = ss

    # 第三次调用：情感选择（先让模型选“情感标签”，再映射到 A/B/C/D）
    choose = data.get("choose", {})
    choose_str = json.dumps(choose, ensure_ascii=False)
    sents_trans_preview = json.dumps(ans_sents, ensure_ascii=False)
    # 规范化 options
    if isinstance(choose, dict):
        options = {str(k).strip(): str(v).strip() for k, v in choose.items()}
    else:
        options = {}

    values = [v for _, v in sorted(options.items())]
    prompt_choose = f"""
你是古诗词情感判断助手。

【诗词】
title：{data.get('title', '')}
author：{data.get('author', '')}
content：{data.get('content', '')}

【逐句译文（供你参考）】
{sents_trans_preview}

【情感选项（只在这些里面选）】
{choose_str}

【任务】
不要输出字母。请只输出最符合整首诗主情感的“情感标签文本”，必须从选项文本中原样选一个。

【输出要求】
1) 只输出严格 JSON。
2) 必须包含且只包含："idx" 和 "emotion_label" 两个字段。
3) emotion_label 必须等于选项里的某个文本（原样）。

输出示例：{{"idx": {idx}, "emotion_label": "{(values[0] if values else '')}"}}

现在开始作答：只输出 JSON。
"""

    choose_text = _call_llm(prompt_choose, max_retries=3)
    choose_json = _extract_json(choose_text) or {}
    emotion_label = ""
    if isinstance(choose_json, dict):
        emotion_label = str(choose_json.get("emotion_label", "")).strip()

    choose_id = ""
    if emotion_label and options:
        # 先精确匹配
        for k, v in options.items():
            if v == emotion_label:
                choose_id = k
                break
        # 再做包含匹配（防止模型多输出少量字）
        if not choose_id:
            for k, v in options.items():
                if v and (v in emotion_label or emotion_label in v):
                    choose_id = k
                    break

    # 兜底：如果映射失败，再让模型直接输出字母
    if choose_id not in {"A", "B", "C", "D"}:
        prompt_choose2 = f"""只输出一个字母（A/B/C/D）表示最符合情感的选项键，不要输出任何其他字符：\n{choose_str}\n"""
        choose_id2 = _call_llm(prompt_choose2, max_retries=2).strip()
        m = re.search(r"[ABCD]", choose_id2)
        choose_id = m.group(0) if m else ""

    return {
        "idx": sents_json.get("idx", idx),
        "ans_qa_words": ans_words,
        "ans_qa_sents": ans_sents,
        "choose_id": choose_id,
    }


def main():
    """批量推理入口：读取评测 JSON，写出 submit.json。"""
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
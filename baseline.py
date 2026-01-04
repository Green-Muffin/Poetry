"""诗词理解与推理评测 Baseline 脚本。

本文件实现一个最小可用的端到端流程：
1）从 JSON 文件读取评测样本；
2）将每条样本发送到 OpenAI 兼容的 Chat 接口（通常由 vLLM 提供服务）；
3）将模型输出解析为 JSON；
4）把结果保存到 `submit.json`。

说明：
- 推理服务需要是 OpenAI 兼容接口（例如：`vllm serve ... --port 8000`）。
- 该 baseline 更关注 IO 格式正确性，而非最优分数。

输出格式（submit.json）：
    [
        {
            "idx": int,
            "ans_qa_words": {"word": "meaning", ...},
            "ans_qa_sents": {"sentence": "translation", ...},
            "choose_id": "A"|"B"|"C"|"D"
        },
        ...
    ]
"""

# 使用 vllm,使用方式见GitHub vllm
# 这是我的服务器启动
# vllm serve /mnt/home/user04/CCL/model/Qwen2.5-7B-Instruct  --served-model-name qwen2.5-7b   --max_model_len 20000
import os
import json
from openai import OpenAI
from tqdm import tqdm
import re

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


def get_response(data):
    """针对单条评测样本请求模型并解析结果。

    参数：
        data (dict)：一条评测样本。常见字段包括 `index`/`idx`、`title`、`author`、`content`、
            `qa_words`、`qa_sents`、`choose`。

    返回：
        dict：从模型输出中解析得到的 JSON 字典；若解析失败，则返回一个字段为空的默认字典。

    行为：
        - 最多重试 3 次；
        - 从模型输出中提取第一个形如 `{...}` 的 JSON 对象进行解析。
    """
    prompt = f"""
        你是一个古诗词专家，现在有一些古诗词需要你的帮助。
        我会给你提供一个 JSON 数据，格式如下：
        - **"index"**：古诗词的序号  
        - **"title"**：古诗词的标题  
        - **"author"**：古诗词的作者  
        - **"content"**：古诗词的内容  
        - **"qa_words"**：古诗词中需要翻译的词语  
        - **"qa_sents"**：古诗词中需要提供白话文译文的句子  
        - **"choose"**：一个包含多个选项的字典，每个选项代表该诗词可能表达的情感  

        这是我的数据：{data}

        ### 你的任务：
        请你根据提供的数据，生成如下 JSON 格式的结果：
        - **"ans_qa_words"**：对 "qa_words" 词语的含义进行解释  
        - **"ans_qa_sents"**：对 "qa_sents" 句子提供白话文译文  
        - **"choose_id"**：从 "choose" 选项中选择最符合该古诗词情感的标号，仅输出对应的字母  

        ### **json输出格式示例：**

        {{
            "idx": 古诗词的序号,
            "ans_qa_words": {{
                "词语1": "词语1的含义",
                "词语2": "词语2的含义",
                "词语3": "词语3的含义"
            }},
            "ans_qa_sents": {{
                "句子1": "句子1的白话文翻译",
                "句子2": "句子2的白话文翻译"
            }},
            "choose_id": ""
        }}

        ### **注意事项：**
        1. **仅返回 JSON 结果，不需要额外解释或输出其他内容。**  
        2. **请确保 "choose_id" 只输出选项的字母，不要附加其他文本。**  
        3. **请保持输出格式整洁，符合 JSON 规范。**
        4.**不要给出任何注释和注意，只需要给出答案，不需要给出其他无关内容**
        5.**请确保输出格式整洁，符合 JSON 规范。**
        6.**必须用中文解答。**
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
                return answer
        except json.JSONDecodeError:
            continue
    return {
        "idx": data.get("index", ""),
        "ans_qa_words": {},
        "ans_qa_sents": {},
        "choose_id": -1
    }


def main():
    """运行 baseline：读取 `data/eval_data.json`，生成并写出 `submit.json`。"""
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
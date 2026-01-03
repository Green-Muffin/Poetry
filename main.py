import os
import json
import re
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# =========================
# 0. 基本环境设置（可选但推荐）
# =========================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================
# 1. 本地模型路径（改成你自己的）
# =========================
MODEL_PATH = "/home/greenmuffin/Repos/PythonProjects/Poetry/models/Qwen/Qwen2.5-7B-Instruct"

# =========================
# 2. 加载 tokenizer
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# =========================
# 3. 4bit 量化配置（关键）
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# =========================
# 4. 加载模型（强制整个模型进 GPU）
# =========================
print("Loading model (4bit, GPU only)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map={"": 0},          # 强制全部放到 GPU 0
    max_memory={0: "5.8GB"},     # 给系统留一点显存余量
    low_cpu_mem_usage=True
)

model.eval()
print("Model loaded.")

# =========================
# 5. 推理函数
# =========================
def get_response(data):
    prompt = f"""
你是一个古诗词专家，现在有一些古诗词需要你的帮助。

我会给你提供一个 JSON 数据，格式如下：
- "index"：古诗词的序号
- "title"：古诗词的标题
- "author"：古诗词的作者
- "content"：古诗词的内容
- "qa_words"：古诗词中需要翻译的词语
- "qa_sents"：古诗词中需要提供白话文译文的句子
- "choose"：一个包含多个选项的字典，每个选项代表该诗词可能表达的情感

这是我的数据：
{data}

### 你的任务：
请你根据提供的数据，生成如下 JSON 格式的结果：
- "ans_qa_words"
- "ans_qa_sents"
- "choose_id"

### JSON 输出示例：
{{
  "idx": {data.get("index", "")},
  "ans_qa_words": {{}},
  "ans_qa_sents": {{}},
  "choose_id": ""
}}

注意：
1. 只输出 JSON
2. 不要输出任何多余文本
3. 必须是合法 JSON
4. 必须使用中文
"""

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.1
            )

        text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())

    except Exception as e:
        print("推理失败：", e)

    finally:
        # 防止显存碎片化
        torch.cuda.empty_cache()

    return {
        "idx": data.get("index", data.get("idx", "")),
        "ans_qa_words": {},
        "ans_qa_sents": {},
        "choose_id": ""
    }

# =========================
# 6. 主流程
# =========================
def main():
    input_path = "data/train-data"
    output_path = "submit.json"

    json_files = []
    for root, _, files in os.walk(input_path):
        for name in files:
            if name.lower().endswith(".json"):
                json_files.append(os.path.join(root, name))
    json_files.sort()

    input_data = []
    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            input_data.extend(data)
        elif isinstance(data, dict):
            input_data.append(data)

    output_data = []

    for i, data in enumerate(tqdm(input_data, desc="Processing")):
        if "index" not in data and "idx" not in data:
            data = dict(data)
            data["index"] = i

        answer = get_response(data)

        idx = answer.get("idx", data.get("index", i))
        output_data.append(
            {
                "idx": idx,
                "ans_qa_words": answer.get("ans_qa_words", {}) or {},
                "ans_qa_sents": answer.get("ans_qa_sents", {}) or {},
                "choose_id": answer.get("choose_id", "") or "",
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"处理完成：共 {len(output_data)} 条，已保存到 {output_path}")

# =========================
# 7. 程序入口
# =========================
if __name__ == "__main__":
    main()

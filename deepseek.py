import json
import os
import time
import torch
from torch import load
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ===================== 核心配置 =====================
DATASET_NAME = "v3t3E1C1"
NUM = 100
TOTAL_COUNT = 100000
START_ID = TOTAL_COUNT - NUM
MODEL_CACHE_DIR = "./DeepSeek-Math-7B-Base"  # 改为你的本地路径

INPUT_JSONL = f"{DATASET_NAME}_deepseek_in.jsonl"
OUTPUT_JSONL = f"{DATASET_NAME}_deepseek_out.jsonl"
LOG_FILE = f"{DATASET_NAME}_deepseek.log"
PT_FILE_PATH = f"../dataset/mul/{DATASET_NAME}/"

# ===================== 1. 自动生成输入文件（完全保留原版） =====================
if not os.path.exists(INPUT_JSONL):
    print(f"📄 生成 {INPUT_JSONL} (最后 {NUM} 条)...\n")
    with open(INPUT_JSONL, "w", encoding="utf-8") as f:
        for idx in range(START_ID, TOTAL_COUNT):
            pt_path = os.path.join(PT_FILE_PATH, f"{idx}.pt")
            d = load(pt_path, weights_only=False)
            # Base模型专用prompt：直接问问题，结尾加"="引导直接输出答案
            prompt = f"({d.system[0]})*({d.system[1]})="
            json_line = {"id": idx, "prompt": prompt, "label": d.product}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    print(f"✅ {INPUT_JSONL} 生成完成！\n")

# ===================== 2. Tokenizer + 模型（DeepSeek专用配置） =====================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CACHE_DIR,
    trust_remote_code=True,
    local_files_only=True,
)
# DeepSeek Base模型的pad_token和eos_token都是<|endoftext|>
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_CACHE_DIR,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",
    # 如果显存不够（<14GB），取消下面一行注释启用4bit量化
    # load_in_4bit=True,
).eval()

# ===================== 3. 推理函数 =====================
# Base 模型用 few-shot 示例引导它输出乘积而非求根
FEW_SHOT_EXAMPLES = (
    "(x+1)*(x+2)=x^2+3*x+2\n"
    "(x-3)*(x+5)=x^2+2*x-15\n"
    "(2*x+1)*(x-4)=2*x^2-7*x-4\n"
    "(3*x-2)*(x+7)=3*x^2+19*x-14\n"
)

# `(` 的 token ID（Base 模型出完答案后会继续生成更多 (expr)*(expr)= 示例
# 把 `(` 作为额外 stop token，模型一开始延续就立即停止，避免浪费算力）
_LPAREN_ID = tokenizer.encode('(', add_special_tokens=False)[0]


def infer(prompt):
    try:
        # few-shot: 示例在前，实际 prompt 在后，模型会按示例格式续写
        full_prompt = FEW_SHOT_EXAMPLES + prompt
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, _LPAREN_ID],
            )
        
        # 取只新生成的 token
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        res = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # byte-level BPE: Ċ = 换行, Ġ = 空格, 需要手动还原
        res = res.replace("Ċ", "\n").replace("Ġ", " ")
        # 只取第一行，去除多余续写
        res = res.strip().split("\n")[0].strip()
        # Base 模型出完答案后会继续生成更多 (expr)*(expr)= 示例，截断到第一个'('
        if '(' in res:
            res = res[:res.index('(')].strip()
        return res
    except Exception as e:
        print(f"推理错误: {e}")
        return ""

# ===================== 4. 推理+可视化+保存 =====================
print("🚀 开始推理（DeepSeek-Math-7B-Base 纯直出，无任何思考）:\n")
total_start = time.time()
times = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out, \
     open(LOG_FILE, "w", encoding="utf-8") as f_log:

    f_log.write(f"Dataset: {DATASET_NAME} | Samples: {NUM}\n")
    f_log.write("=" * 60 + "\n")

    for index, line in enumerate(f_in, 1):
        line = line.strip()
        if not line: continue
        
        data = json.loads(line)
        t0 = time.time()
        qwen_output = infer(data["prompt"])
        elapsed = time.time() - t0
        data["qwen"] = qwen_output
        data["time_s"] = round(elapsed, 2)
        times.append(elapsed)

        print("=" * 80)
        print(f"第 {index} 条 | ID: {data['id']} | {elapsed:.2f}s")
        print(f"Prompt: {data['prompt']}")
        print(f"标准答案: {data['label']}")
        print(f"DeepSeek输出: {qwen_output}")
        print("=" * 80 + "\n")

        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    total_elapsed = time.time() - total_start
    avg = total_elapsed / max(len(times), 1)
    f_log.write("=" * 60 + "\n")
    f_log.write(f"Total: {total_elapsed:.2f}s | Avg: {avg:.2f}s/sample\n")

print(f"🎉 完成：DeepSeek-Math-7B-Base 纯公式直出，无任何思考过程")
print(f"⏱  总耗时: {total_elapsed:.2f}s | 平均: {avg:.2f}s/条")
print(f"📝 耗时日志: {LOG_FILE}")
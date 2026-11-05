import json
import os
import time
import torch
from torch import load
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ===================== 核心配置（完全你的原版） =====================
DATASET_NAME = "v3t3E1C1"
NUM = 100
TOTAL_COUNT = 100000
START_ID = TOTAL_COUNT - NUM
MODEL_CACHE_DIR = "./Qwen3.5-4B"

INPUT_JSONL = f"{DATASET_NAME}_qwen_nothink_in.jsonl"
OUTPUT_JSONL = f"{DATASET_NAME}_qwen_nothink_out.jsonl"
LOG_FILE = f"{DATASET_NAME}_qwen_nothink.log"
PT_FILE_PATH = f"../dataset/mul/{DATASET_NAME}/"

# ===================== 1. 自动生成输入文件 =====================
if not os.path.exists(INPUT_JSONL):
    print(f"📄 生成 {INPUT_JSONL} (最后 {NUM} 条)...\n")
    with open(INPUT_JSONL, "w", encoding="utf-8") as f:
        for idx in range(START_ID, TOTAL_COUNT):
            pt_path = os.path.join(PT_FILE_PATH, f"{idx}.pt")
            d = load(pt_path, weights_only=False)
            prompt = f"Please directly yield the final polynomial product without thinking:({d.system[0]})*({d.system[1]})"
            json_line = {"id": idx, "prompt": prompt, "label": d.product}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    print(f"✅ {INPUT_JSONL} 生成完成！\n")

# ===================== 2. Tokenizer + 模型 完全原版不动 =====================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CACHE_DIR,
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_CACHE_DIR,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto"
).eval()

# ===================== 3. 推理函数（禁用思考，直出答案） =====================
def infer(prompt):
    try:
        messages = [
            {"role": "user", "content": prompt},
        ]
        # Qwen3.x: enable_thinking=False 跳过思考链，直接输出答案
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        res = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        res = res.strip()
        return res
    except:
        return ""

# ===================== 4. 推理+可视化+保存 =====================
print("🚀 开始推理（底层禁用思考，原生直出答案）:\n")
total_start = time.time()
times = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out, \
     open(LOG_FILE, "w", encoding="utf-8") as f_log:

    f_log.write(f"Dataset: {DATASET_NAME} | Samples: {NUM} | Mode: nothink\n")
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
        print(f"Qwen输出: {qwen_output}")
        print("=" * 80 + "\n")

        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    total_elapsed = time.time() - total_start
    avg = total_elapsed / max(len(times), 1)
    f_log.write("=" * 60 + "\n")
    f_log.write(f"Total: {total_elapsed:.2f}s | Avg: {avg:.2f}s/sample\n")

print(f"🎉 完成：禁用思考模式，纯公式答案")
print(f"⏱  总耗时: {total_elapsed:.2f}s | 平均: {avg:.2f}s/条")
print(f"📝 耗时日志: {LOG_FILE}")
'''
    Conduct polynomial multiplication by calling DeepSeek Math 7B locally deployed
    Coded and annotated by Yuxuan SONG, Bin WANG
    Last modified time: 5 Nov 2026
'''
import json
import os
import time
import torch
from torch import load
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

DATASET_NAME = "v3t3E1C1"
NUM = 100
TOTAL_COUNT = 100000
START_ID = TOTAL_COUNT - NUM
MODEL_CACHE_DIR = "./DeepSeek-Math-7B-Base"

INPUT_JSONL = f"{DATASET_NAME}_deepseek_in.jsonl"
OUTPUT_JSONL = f"{DATASET_NAME}_deepseek_out.jsonl"
LOG_FILE = f"{DATASET_NAME}_deepseek.log"
PT_FILE_PATH = f"../dataset/mul/{DATASET_NAME}/"

# ===================== 1. Input file processing =====================
if not os.path.exists(INPUT_JSONL):
    print(f"Generating {INPUT_JSONL} (last {NUM} lines)...\n")
    with open(INPUT_JSONL, "w", encoding="utf-8") as f:
        for idx in range(START_ID, TOTAL_COUNT):
            pt_path = os.path.join(PT_FILE_PATH, f"{idx}.pt")
            d = load(pt_path, weights_only=False)
            # Ask question directly
            prompt = f"({d.system[0]})*({d.system[1]})="
            json_line = {"id": idx, "prompt": prompt, "label": d.product}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    print(f"{INPUT_JSONL} Finish!\n")

# ===================== 2. Tokenizer + LLM loading =====================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CACHE_DIR,
    trust_remote_code=True,
    local_files_only=True,
)
# The padding and EoS tokens are both '<|endoftext|>'
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_CACHE_DIR,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",
    # load_in_4bit=True,
).eval()

# ===================== 3. Inference function =====================
# Few shot hints
FEW_SHOT_EXAMPLES = (
    "(x+1)*(x+2)=x^2+3*x+2\n"
    "(x-3)*(x+5)=x^2+2*x-15\n"
    "(2*x+1)*(x-4)=2*x^2-7*x-4\n"
    "(3*x-2)*(x+7)=3*x^2+19*x-14\n"
)

# Base models tend to keep generating more (expr)*(expr)= examples after finishing the answer. 
# Using `(` as an extra stop token immediately halts generation once the model starts to continue.
_LPAREN_ID = tokenizer.encode('(', add_special_tokens=False)[0] # Token ID for `(`


def infer(prompt):
    try:
        # few-shot
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
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        res = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # byte-level BPE: Ċ = '\n', Ġ = ' '
        res = res.replace("Ċ", "\n").replace("Ġ", " ")
        # Extract the first line
        res = res.strip().split("\n")[0].strip()
        # Truncate response
        if '(' in res:
            res = res[:res.index('(')].strip()
        return res
    except Exception as e:
        print(f"Error: {e}")
        return ""

# ===================== 4. Inference =====================
print("DeepSeek-Math-7B-Base inferences with no-think module\n")
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
        print(f"{index} | ID: {data['id']} | {elapsed:.2f}s")
        print(f"Prompt: {data['prompt']}")
        print(f"Answer: {data['label']}")
        print(f"DeepSeek output: {qwen_output}")
        print("=" * 80 + "\n")

        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    total_elapsed = time.time() - total_start
    avg = total_elapsed / max(len(times), 1)
    f_log.write("=" * 60 + "\n")
    f_log.write(f"Total: {total_elapsed:.2f}s | Avg: {avg:.2f}s/sample\n")

print(f"Finish: DeepSeek-Math-7B-Base no think module")
print(f"Total elapsed time: {total_elapsed:.2f}s | Avg: {avg:.2f}s")
print(f"Logging file: {LOG_FILE}")
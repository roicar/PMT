# File organization in other directories
See `README.md` in corresponding directories 
# File organization of the base directory
## deepseek.py
Conduct polynomial multiplication by calling DeepSeek Math 7B locally deployed
## evaluate.py
Evaluate extracted LLM predictions using project metrics
## extract.py
Extract final answer from Qwen output and normalize to label format
## preview_tokenized.py
View tokenized collect data: input (f<op>g) and output (collected product)
## qwen_deepthink.py
Conduct polynomial multiplication by calling Qwen 3.5 4B locally deployed with deep think module
## qwen_nothink.py
Conduct polynomial multiplication by calling Qwen 3.5 4B locally deployed with no think module
## run.py
Main entry of the program
## run.sh
Batched scripts ready for running
## scheme3_benckmark.py
Evaluate Scheme 3 (Collect-to-Accum) on 100 samples per dataset for Table 6 comparison
## requirements.txt
Required python libraries
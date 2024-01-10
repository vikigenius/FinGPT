import sys
sys.path.append('/home/void/FinNLP')

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast   # 4.30.2
from peft import PeftModel  # 0.4.0
import torch

from finnlp.benchmarks.fpb import test_fpb
from finnlp.benchmarks.fiqa import test_fiqa , add_instructions
from finnlp.benchmarks.tfns import test_tfns
from finnlp.benchmarks.nwgi import test_nwgi

base_model = "base_models/Llama-2-13b-hf" 
peft_model = "finetuned_models"
tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map = "cuda:0", load_in_8bit = True,)
model = PeftModel.from_pretrained(model, peft_model)
model = torch.compile(model)  # Please comment this line if your platform does not support torch.compile
model = model.eval()

batch_size = 16

res = test_fpb(model, tokenizer, batch_size = batch_size)

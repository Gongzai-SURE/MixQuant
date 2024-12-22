from eetq import AutoEETQForCausalLM
from transformers import AutoTokenizer

model_name = '/model/llama3-8b'
quant_path = "/model/llama3-8b-eetq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoEETQForCausalLM.from_pretrained(model_name)
model.quantize(quant_path)
tokenizer.save_pretrained(quant_path)
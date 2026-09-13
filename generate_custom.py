import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_WEIGHTS = "./joker_lora_weights"

tokenizer = AutoTokenizer.from_pretrained(LORA_WEIGHTS)
# Загружаем чистую базовую модель
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
# Накатываем поверх наши обученные веса анекдотов
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)

prompt = "Заходит программист в столовую, а там"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.85)
print("\n--- Шутка от дообученной модели ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

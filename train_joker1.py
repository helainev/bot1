# Задача вять готовую русскоязычную модель (до 6 ГБ) и дообучить её самостоятельно. 
# Для этого есть специальный датасет с русскими анекдотами: samedad/mem-and-russian-jokes-dataset .
# берем Qwen/Qwen2.5-Coder-3B-Instruct и дообучить её на этом датасете с помощью LoRA. 
# дообучение (Fine-Tuning) локальной модели до 6 ГБ на компьютере с RTX 4060 Ti (8 ГБ VRAM) — это абсолютно выполнимая задача.
# Чтобы модель полностью поместилась в память вместе с градиентами во время обучения, 
# применим технологию QLoRA (квантование в 4 бит + низкоранговая адаптация) 
# [RichardErkhov/igorktech_-_rugpt3-joker-150k-4bits].
# Обучение займет около 15 часов, а на выходе получим свою кастомную нейросеть, которая будет шутить именно так, как нужно.

import torch
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig  # Добавили SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# 1. Настройки моделей и датасета
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"  # Отличная база до 6 ГБ
DATASET_NAME = "samedad/mem-and-russian-jokes-dataset"
OUTPUT_DIR = "./joker_lora_weights"

print("📥 Загрузка датасета и токенизатора...")
dataset = load_dataset(DATASET_NAME, split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 2. Конфигурация 4-битного квантования (чтобы влезть в 8 ГБ VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("🤖 Загрузка базовой модели в 4 битах...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Подготовка модели к обучению в пониженной точности
model = prepare_model_for_kbit_training(model)

# 3. Настройка параметров LoRA
lora_config = LoraConfig(
    r=16,                          # Размерность адаптера
    lora_alpha=32,                 # Коэффициент масштабирования
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Слои для обучения
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Параметры обучения (оптимизировано под RTX 4060 Ti)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,      # Маленький батч для экономии VRAM
    gradient_accumulation_steps=4,      # Виртуально увеличиваем батч до 8
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,                 # 1 эпоха для теста
    bf16=True,                          # Быстрые вычисления на RTX 40-й серии
    optim="paged_adamw_8bit",           # Экономичный по памяти оптимизатор
    save_strategy="epoch",
    report_to="none",
    max_length=512,                 
)

# 5. Запуск процесса дообучения
print("🚀 Запуск обучения...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,                 # Передаем наш SFTConfig
)

trainer.train()

# Сохраняем обученные веса LoRA
print("💾 Сохранение обученного адаптера...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Обучение успешно завершено!")

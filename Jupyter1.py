import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Указываем репозиторий с 4-битной версией модели joker-150k
model_name = "RichardErkhov/igorktech_-_rugpt3-joker-150k-4bits"

print("Загрузка модели анекдотов...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

print("\n🎉 Модель успешно загружена и готова шутить!")
print("Чтобы выйти из генератора, нажмите Ctrl + C или введите 'выход'\n")

# Бесконечный цикл для интерактива в терминале
while True:
    try:
        # Спрашиваем начало анекдота у пользователя
        user_prompt = input("Введите начало анекдота >>> ")
        
        # Проверка на выход
        if user_prompt.lower().strip() in ["выход", "exit", "quit"]:
            print("Работа завершена. До встречи!")
            break
            
        if not user_prompt.strip():
            print("Начало анекдота не может быть пустым. Попробуйте еще раз.")
            continue

        print("Генерация ответа...")
        
        # Токенизация и отправка на GPU
        inputs = tokenizer(user_prompt, return_tensors="pt").to("cuda")
        
        # Генерация текста
        outputs = model.generate(
            **inputs, 
            max_length=120, 
            do_sample=True, 
            temperature=0.92, 
            top_k=50,
            repetition_penalty=1.1  # Штраф за повторение одних и тех же слов
        )
        
        # Декодирование и вывод результата
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n--- Результат генерации ---")
        print(result)
        print("-" * 27 + "\n")
        
    except KeyboardInterrupt:
        print("\nРабота завершена. До встречи!")
        break
    except Exception as e:
        print(f"Произошла ошибка: {e}\n")

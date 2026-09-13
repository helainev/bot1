import streamlit as st
import requests
import json

# Настройка страницы
st.set_page_config(page_title="Локальный ИИ-Чат", page_icon="🤖", layout="wide")

# Константы для подключения к Ollama API
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# Боковая панель для настроек
with st.sidebar:
    st.header("⚙️ Настройки")
    # Выпадающий список с моделями, которые у вас гарантированно скачаны
    selected_model = st.selectbox(
        "Выберите модель нейросети:",
        ["mistral-nemo", "gemma4:12b"],
        index=0
    )
    
    # Кнопка очистки истории
    if st.button("🗑 Очистить историю чата"):
        st.session_state.messages = []
        st.rerun()

st.title(f"🤖 Чат-бот (Текущая модель: {selected_model})")

# Инициализация истории сообщений в сессии
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Привет! Я готов к работе. Задай мне любой вопрос!"}
    ]

# Отображение истории чата
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Поле ввода реплики
if user_input := st.chat_input("Введите ваше сообщение..."):
    # Сохраняем и выводим сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Область вывода ответа ассистента
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # Передаем выбранную в Sidebar модель и всю историю диалога
        payload = {
            "model": selected_model,
            "messages": st.session_state.messages,
            "stream": True
        }
        
        try:
            response = requests.post(OLLAMA_URL, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    token = chunk.get("message", {}).get("content", "")
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except requests.exceptions.RequestException as e:
            placeholder.error(f"❌ Ошибка подключения к Ollama: {e}\nУбедитесь, что служба запущена.")

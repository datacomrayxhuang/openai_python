import os
import streamlit as st
from openai import OpenAI

# Get api key and base url from .env file
openai_base_url = os.environ.get("OPENAI_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Basic UI set up
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# OpenAI client
client = OpenAI(
    base_url = openai_base_url,
    api_key = openai_api_key,
)

if prompt := st.chat_input():
    try:
        # Record user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display the input on UI
        st.chat_message("user").write(prompt)
        # Call OpenAI API with history
        response = client.chat.completions.create(
            model = "Gpt4o",
            messages = st.session_state.messages,
        )
        # Get response
        result = response.choices[0].message.content
        # Record response
        st.session_state.messages.append({"role": "assistant", "content": result})
        # Write response to history
        st.chat_message("assistant").write(result)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

import os
import streamlit as st
import chromadb
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletionMessage

# Persistent database
chroma_client = chromadb.PersistentClient(path = "./db/task2/chroma")
collection = chroma_client.get_collection(name = 'task2')

# Get api key and base url from .env file
openai_base_url = os.environ.get("OPENAI_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Basic UI set up
st.title("📝 Front-end Innovation Q&A with OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You're a helpful assistant which answer questions about Front-end Innovation team."},
        {"role": "assistant", "content": "Ask something about the Front-end Innovation team"}
    ]

for message in st.session_state.messages:
    if type(message) == ChatCompletionMessage:
        message = message.model_dump()
    if message["role"] != "system":
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

        # Query local data
        results = collection.query(
            query_texts = [prompt],
            n_results = 20,
        )
        documents = collection.get(ids = results['ids'][0])['documents']
        context = ''
        for chunk in documents:
            context += f"""{chunk}\n\n"""

        # Instruction to LLM, specifically ask LLM to answer only using context provided
        message = f"Answer the question using only the context provided.\n\nQuestion: {prompt}\nContext:\n\n{context}\n"
        # Record system instruction
        st.session_state.messages.append({"role": "system", "content": message})

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

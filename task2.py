import os
import streamlit as st
import chromadb
from openai import OpenAI

chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_collection(name = 'task2')

# Get api key and base url from .env file
openai_base_url = os.environ.get("OPENAI_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")

st.title("📝 Front-end Innovation Q&A with OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "You're a helpful assistant which answer questions about Front-end Innovation team."}]
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the Front-end Innovation team"}]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

client = OpenAI(
    base_url = openai_base_url,
    api_key = openai_api_key,
)

if prompt := st.chat_input():
    # Record user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the input on UI
    st.chat_message("user").write(prompt)

    results = collection.query(
        query_texts = [prompt],
        n_results = 20,
    )
    documents = collection.get(ids = results['ids'][0])['documents']
    context = ''
    for chunk in documents:
        context += f"""{chunk}\n\n"""
    message = f"""Answer the question using only the context provided.\n\nQuestion: {prompt}\n\Context:\n\n{context}\n"""
    # Record system query result
    st.session_state.messages.append({"role": "system", "content": message})
    # Display the system query result on UI
    # st.chat_message("system").write(message)

    # Call OpenAI API with history
    response = client.chat.completions.create(
        model = "Gpt4o",
        messages = st.session_state.messages,
    )
    # Get response
    message = response.choices[0].message.content
    # Record response
    st.session_state.messages.append({"role": "assistant", "content": message})
    # Write response to history
    st.chat_message("assistant").write(message)

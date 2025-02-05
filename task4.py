import os
import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# Get api key and base url from .env file
openai_base_url = os.environ.get("OPENAI_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Basic UI set up
st.title("📝 Dynamic Document Generator with OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are an automated dynamic document generator assistant"},
        {"role": "assistant", "content": "Ask me to generate a document for you."},
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

# Structured output for first LLM call: determine if the input is a document event
class EventExtraction(BaseModel):
    description: str = Field(description = "Raw description of the latest user input")
    is_new_document_event: bool = Field(description = "Whether the latest user input describes a new document generating event")
    is_change_document_event: bool = Field(description = "Whether the latest user input describes a change event for a document generated")
    confidence_score: float = Field(description = "Confidence score between 0 and 1")

# Structured output for extracting specific requirements
class NewDocumentEvent(BaseModel):
    description: str = Field(description = "Raw description of the event")
    document_length: int = Field(description = "Does this text require a document generated to have a specific length, if no, answer any")
    document_style: str = Field(description = "Does this text require a document generated to have a specific style, if no, answer any")
    key_words: list[str] = Field(description = "What are they keywords or bullet points user requested, if none, answer none")
    confidence_score: float = Field(description = "Confidence score between 0 and 1")

# Structured output for second LLM call
class ComparisonResult(BaseModel):
    satisfy_requirement: bool = Field(description = "Does the draft generated satisfy the requirement")
    unsatisfy_reason: str = Field(description = "If the draft generated does not satisfy the requirement, what is the reason? Is satisfy, answer None")
    confidence_score: float = Field(description = "Confidence score between 0 and 1")

# First LLM call to determine if input is a document generating event
def analyse_input(messages: list[ChatCompletionMessageParam]) -> EventExtraction:
    messages = messages.copy()
    messages.append({"role": "system", "content": f"Analyse if the latest user input describes a new document generating event or a change event for a document generated, cannot be both true at the same time."})
    completion = client.beta.chat.completions.parse(
        model = "Gpt4o",
        messages = messages,
        response_format = EventExtraction,
    )
    result = completion.choices[0].message.parsed
    return result

# Second LLM call to extract specific requirements in user input
def extract_requirements(user_input: str) -> NewDocumentEvent:
    message = [{"role": "system", "content": f"Extract all the specific requirements in the given the user input: {user_input}."}]
    completion = client.beta.chat.completions.parse(
        model = "Gpt4o",
        messages = message,
        response_format = NewDocumentEvent,
    )
    result = completion.choices[0].message.parsed
    return result

if prompt := st.chat_input():
    try:
        # Step 1 record user input and analyse it
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Step 2 analyse user input
        result = analyse_input(st.session_state.messages)
        if (result.is_new_document_event == False and result.is_change_document_event == False) or result.confidence_score < 0.7:
            st.info(f"The input is not a document event.")
            st.stop()

        # Step 3 extract requirements
        if result.is_new_document_event:
            requirements = extract_requirements(prompt)
            st.session_state.messages.append({"role": "system", "content": f"Given the user input: {prompt}\n\nGenerate document, make sure its length satisfy \"{requirements.document_length}\", its style is in \"{requirements.document_style}\" and includes keywords of \"{requirements.key_words}\""})
        
        # Step 4 - Third LLM call to generate the draft
        completion = client.chat.completions.create(
            model = "Gpt4o",
            messages = st.session_state.messages,
        )
        draft = completion.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": draft})
        st.chat_message("assistant").write(draft)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

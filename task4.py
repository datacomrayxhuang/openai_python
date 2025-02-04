import os
import json
import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI

# Get api key and base url from .env file
openai_base_url = os.environ.get("OPENAI_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")

st.title("📝 Dynamic Document Generator with OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are an automated dynamic document generator assistant"},
        {"role": "assistant", "content": "Ask me to generate a document for you."},
    ]
if "disabled" not in st.session_state:
    st.session_state.disabled = False

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

client = OpenAI(
    base_url = openai_base_url,
    api_key = openai_api_key,
)

class EventExtraction(BaseModel):
    description: str = Field(description = "Raw description of the event")
    is_document_generating_event: bool = Field(description = "Whether this text describes a document generating request?")
    target_length: int = Field(description = "Does this text require a document generated to have a specific length?")
    document_style: str = Field(description = "Does this text require a document generated to have a specific style?")
    key_words: list[str] = Field(description = "What are they keywords or bullet points user requested when generating a document")
    confidence_score: float = Field(description = "Confidence score between 0 and 1")

class ComparisonResult(BaseModel):
    satisfyRequirement: bool = Field(description = "Does the draft generated satisfy the requirement")
    confidence_score: float = Field(description = "Confidence score between 0 and 1")

# First LLM call to determine if input is a document generating event
def analyse_input(user_input: str) -> EventExtraction:
    completion = client.beta.chat.completions.parse(
        model = "Gpt4o",
        messages = [
            {
                "role": "system",
                "content": f"Analyze if the text describes a document generating event.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format = EventExtraction,
    )
    result = completion.choices[0].message.parsed
    return result

# Second LLM call to make sure the draft satisfy the requirements
def compare_with_requirement(requirement: str, draft: str) -> ComparisonResult:
    message = f"Given the requirement \"{requirement}\", dose the draft generated satisfy the requirements?\n\nDraft: {draft}\n\n"
    completion = client.beta.chat.completions.parse(
        model = "Gpt4o",
        messages = [
            {
                "role": "system",
                "content": message,
            },
        ],
        response_format = ComparisonResult,
    )
    result = completion.choices[0].message.parsed
    return result

# Third LLM call to fine tune the draft
def fine_tune_draft(requirement: str, draft: str) -> str | None:
    message = f"Given the requirement \"{requirement}\", fine tune the draft generated so that it satisfies the requirements\n\nDraft: {draft}\n\n"
    completion = client.chat.completions.create(
        model = "Gpt4o",
        messages = [
            {
                "role": "system",
                "content": message,
            },
        ],
    )
    result = completion.choices[0].message.content
    return result

def disable():
    st.session_state.disabled = True

prompt = st.chat_input(disabled = st.session_state.disabled, on_submit = disable)

if prompt != None and prompt != '':
    st.chat_message("user").write(prompt)

    st.chat_message("assistant").write("Analysing input.")
    result = analyse_input(prompt)
    if result.is_document_generating_event == False or result.confidence_score < 0.7:
        st.info("The input is not a document generating request.")
        st.stop()
    
    st.session_state.messages.append({"role": "system", "content": json.dumps(result.model_dump())})
    st.session_state.messages.append({"role": "system", "content": "Now complete user's request"})
    
    completion = client.chat.completions.create(
        model = "Gpt4o",
        messages = st.session_state.messages,
    )
    draft = completion.choices[0].message.content
    # st.chat_message("assistant").write(result)

    max_steps = 3
    step = 1
    
    while draft != None and step <= max_steps:
        st.chat_message("assistant").write("Making sure the draft meets requirements.")
        comparison_result = compare_with_requirement(prompt, draft)
        
        if comparison_result.satisfyRequirement and comparison_result.confidence_score > 0.9:
            break
        else:
            st.chat_message("assistant").write("Fine tuning the draft.")
            new_draft = fine_tune_draft(prompt, draft)
            step += 1
            if new_draft != None:
                draft = new_draft
    
    st.session_state.messages.append({"role": "assistant", "content": draft})
    st.chat_message("assistant").write(draft)

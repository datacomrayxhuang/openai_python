import os
from dotenv import load_dotenv
import json
import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from subprocess import run

# Get api key and base url from .env file
load_dotenv()
openai_base_url = os.getenv("OPENAI_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Basic UI set up
st.title("📝 OpenAI Python Coding Assistant")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "You are a Python AI coding assistant, you only know Python. When generating any code, always make the code excecutable and includes different inputs as examples. Print statements should have new line at the end."}]

# Structured output for first LLM call: determine if the input is a document event
class EventExtraction(BaseModel):
    description: str = Field(description = "Raw description of the latest user input")
    generated_code: str = Field(description = "Any excecutable Python code generated in this answer")
    general_output: str = Field(description = "Answer which will be displayed to the user, including any code generated")
    confidence_score: float = Field(description = "Confidence score between 0 and 1")

for message in st.session_state.messages:
    if type(message) == ChatCompletionMessage:
        message = message.model_dump()
    elif type(message) != dict:
        try:
            message = message.model_dump()
        except Exception as e:
            st.error(f"Error: {e}")
    
    if type(message) == dict:
        if message["role"] != "system" and message["role"] != "tool" and message["content"] != None:
            try:
                parsed = json.loads(message["content"])
                general_output = parsed["general_output"]
                if general_output != None:
                    st.chat_message(message["role"]).write(general_output)
            except Exception as e:
                st.chat_message(message["role"]).write(message["content"])

# OpenAI client
client = OpenAI(
    base_url = openai_base_url,
    api_key = openai_api_key,
)

# Tools available to LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_local_code",
            "description": "Read code generated and stored in local.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

# First tool LLM can call to get code from output.py
def get_local_code() -> str:
    with open("output.py", "r") as file:
        code = file.read()
        file.close()
    return code

# Helper function to make tool call
def call_function(name, args):
    if name == "get_local_code":
        return get_local_code(**args)

def make_tool_calls(tool_calls: list[ChatCompletionMessageToolCall]):
    step_messages = []

    for tool_call in tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        step_message = f"Step: {name}"

        step_messages.append(step_message)
        st.chat_message("assistant").write(step_message)

        function_result = call_function(name, args)
        
        st.session_state.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(function_result)})

    for step_message in step_messages:
        st.session_state.messages.append({"role": "assistant", "content": step_message})

def handle_result(result: EventExtraction):
    st.session_state.messages.append({"role": "assistant", "content": result.general_output})
    st.chat_message("assistant").write(result.general_output)
    if result.generated_code != '':
        with open("output.py", "w") as file:
            file.write(result.generated_code)
            file.close()
        try:
            if os.path.exists("./output.py"):
                os.chmod("./output.py", 0o755)
                st.info(f"Excecuting code...")
                output = run(["python", "output.py"], capture_output = True, timeout = 30).stdout.decode("utf-8")
                st.info(f"Finished running")
                st.info(f"You may find the code in output.py.\n\nExcecuted code, output:\n\n{output}")
            else:
                print("File not found:", "./output.py")
        except PermissionError:
            print("Permission denied: You don't have the necessary permissions to change the permissions of this file.")

def call_llm() -> ParsedChatCompletion[EventExtraction]:
    completion = client.beta.chat.completions.parse(
        model = "Gpt4o",
        messages = st.session_state.messages,
        response_format = EventExtraction,
        tools = tools,
    )
    result = completion.choices[0].message.parsed
    if result != None:
        handle_result(result = result)

    return completion

if prompt := st.chat_input():
    try:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        completion = call_llm()
        
        if completion.choices[0].message.tool_calls != None and len(completion.choices[0].message.tool_calls) > 0:
            st.session_state.messages.append(completion.choices[0].message)
            make_tool_calls(completion.choices[0].message.tool_calls)
            call_llm()
        
        if completion.choices[0].message.tool_calls != None and len(completion.choices[0].message.tool_calls) == 0:
            completion.choices[0].message.tool_calls = None
            st.session_state.messages.append(completion.choices[0].message)

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

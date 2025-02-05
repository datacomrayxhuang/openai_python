import os
import json
import streamlit as st
import chromadb
from pydantic import BaseModel, Field
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

# Persistent database
chroma_client = chromadb.PersistentClient(path = "./db/task3/chroma")
collection = chroma_client.get_collection(name = 'task3')

# Get api key and base url from .env file
openai_base_url = os.environ.get("OPENAI_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Basic UI set up
st.title("📝 Trip planner with OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are an automated trip planning assistant"},
        {"role": "assistant", "content": "Ask me to plan a trip for you."},
    ]

for message in st.session_state.messages:
    if type(message) == ChatCompletionMessage:
        message = message.model_dump()
    if message["role"] != "system" and message["role"] != "tool" and message["content"] != None:
        st.chat_message(message["role"]).write(message["content"])

# OpenAI client
client = OpenAI(
    base_url = openai_base_url,
    api_key = openai_api_key,
)

locally_stored_documents = [
    "auckland_attraction.pdf",
    "hamilton_waikato_attraction.pdf",
    "rotorua_attraction.pdf",
    "taupo_attraction.pdf",
    "tauranga_bay_of_plenty_attraction.pdf",
]

# Structured output for first LLM call: determine if the input is a trip planning event
class EventExtraction(BaseModel):
    description: str = Field(description = "Raw description of the event")
    is_trip_planning_event: bool = Field(description = "Whether this text describes a trip planning event")
    location: list[str] = Field(description = "List of location of the event")
    should_query_local_data_for_location: list[bool] = Field(description = f"Given local data: {locally_stored_documents}, in the same order of list of location of the event, whether query local data for a location")
    trip_duration: str = Field(description = "Duration of the trip, if unknown, answer unknown")
    confidence_score: float = Field(description = "Confidence score between 0 and 1")

# Structured output for second LLM call
class Attraction(BaseModel):
    name: str = Field(description = "Name of the attraction")
    description: str = Field(description = "Description of the attraction")
    duration: str = Field(description = "Duration of the attraction")

# Structured output for second LLM call
class AttractionExtraction(BaseModel):
    attractions: list[Attraction] = Field(description = "List of attractions in this location")

# Tools available to LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "analyse_input",
            "description": "Analyse if the text describes a trip planning event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {"type": "string"},
                },
                "required": ["user_input"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attractions",
            "description": "Get attractions for a certain location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "should_query_local_data": {"type": "boolean"},
                },
                "required": ["location", "should_query_local_data"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_itinerary",
            "description": "Generate itinerary using a list of attractions and duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "attractions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "duration": {"type": "string"},
                            },
                            "required": ["name", "description", "duration"],
                            "additionalProperties": False,
                        },
                    },
                    "duration": {"type": "string"},
                },
                "required": ["attractions", "duration"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trip_summary",
            "description": "Provide a summary for a trip planning event using itinerary provided",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {"type": "string"},
                    "itinerary": {"type": "string"},
                },
                "required": ["user_input", "itinerary"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

# First tool LLM can call to determine if input is a trip planning event
def analyse_input(user_input: str) -> EventExtraction:
    completion = client.beta.chat.completions.parse(
        model = "Gpt4o",
        messages = [
            {"role": "system", "content": f"Analyse if the text describes a trip planning event."},
            {"role": "user", "content": user_input},
        ],
        response_format = EventExtraction,
    )
    result = completion.choices[0].message.parsed
    return result

# Second tool LLM can call to get attractions for a location (may or may not use local data)
def get_attractions(location: str, should_query_local_data: bool) -> AttractionExtraction:
    message = ''
    if should_query_local_data:
        local_data = query_local_data(location)
        message = f"Get attractions in this location: {location}, prioritise using the following local data: {local_data}."
    else:
        message = f"Get attractions in this location: {location}."
    
    completion = client.beta.chat.completions.parse(
        model = "Gpt4o",
        messages = [{"role": "system", "content": message}],
        response_format = AttractionExtraction,
    )
    result = completion.choices[0].message.parsed
    return result

# Third tool LLM can call to generate itinerary
def generate_itinerary(attractions: list[Attraction], duration: str) -> str:
    message = ''
    if duration != 'unknown':
        message = f"Generate an itinerary for the following attractions: {attractions}, spread into {duration}."
    else:
        message = f"Generate an itinerary for the following attractions: {attractions}, spread into suitable days."
    
    completion = client.chat.completions.create(
        model = "Gpt4o",
        messages = [{"role": "system", "content": message}],
    )
    result = completion.choices[0].message.content
    return result

# Forth tool LLM can call to generate itinerary
def get_trip_summary(user_input: str, itinerary: str) -> str:
    message = f"Using the itinerary: {itinerary}.\n\nSummarise a natural language response to the user's goal: {user_input}."

    completion = client.chat.completions.create(
        model = "Gpt4o",
        messages = [{"role": "system", "content": message}],
    )
    result = completion.choices[0].message.content
    return result

# Optional to query local data
def query_local_data(location: str) -> str:
    local_data = ''
    query_results = collection.query(
        query_texts = [location],
        n_results = 20,
    )
    documents = collection.get(ids = query_results['ids'][0])['documents']
    for chunk in documents:
        local_data += f"""{chunk}\n\n"""
    return local_data

# Helper function to make tool call
def call_function(name, args):
    if name == "analyse_input":
        return analyse_input(**args)
    elif name == "get_attractions":
        return get_attractions(**args)
    elif name == "generate_itinerary":
        return generate_itinerary(**args)
    elif name == "get_trip_summary":
        return get_trip_summary(**args)

def make_tool_calls(tool_calls: list[ChatCompletionMessageToolCall]):
    st.session_state.messages.append(completion.choices[0].message)
    step_messages = []

    for tool_call in tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        step_message = ''
        if name == 'get_attractions' and args['should_query_local_data']:
            step_message = f"Step: {name} for {args['location']} (use local data)"
        elif name == 'get_attractions' and not args['should_query_local_data']:
            step_message = f"Step: {name} for {args['location']}"
        else:
            step_message = f"Step: {name}"

        step_messages.append(step_message)
        st.chat_message("assistant").write(step_message)

        function_result = call_function(name, args)
        # st.chat_message("system").write(function_result)

        if type(function_result) == EventExtraction:
            function_result = function_result.model_dump()
        elif type(function_result) == AttractionExtraction:
            function_result = function_result.model_dump()
        
        st.session_state.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(function_result)})

    for step_message in step_messages:
        st.session_state.messages.append({"role": "assistant", "content": step_message})

if prompt := st.chat_input():
    try:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        sys_message = f"User wants to achieve the goal of {prompt}. Start with calling analyse_input if you haven't, break it down into steps, in each step, only use tools provided."
        st.session_state.messages.append({"role": "system", "content": sys_message})
        completion = client.chat.completions.create(
            model = "Gpt4o",
            messages = st.session_state.messages,
            tools = tools,
        )
        result = completion.choices[0].message.content

        max_steps = 20
        step = 1
        
        while result == None and step <= max_steps:
            completion = client.chat.completions.create(
                model = "Gpt4o",
                messages = st.session_state.messages,
                tools = tools,
            )
            step += 1
            result = completion.choices[0].message.content
            if completion.choices[0].message.tool_calls != None:
                make_tool_calls(completion.choices[0].message.tool_calls)
        
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").write(result)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

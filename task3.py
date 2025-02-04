import os
import json
import streamlit as st
import chromadb
from pydantic import BaseModel, Field
from openai import OpenAI

chroma_client = chromadb.PersistentClient(path = "./db/task3/chroma")
collection = chroma_client.get_collection(name = 'task3')

# Get api key and base url from .env file
openai_base_url = os.environ.get("OPENAI_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")

st.title("📝 Trip planner with OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are an automated trip planning assistant"},
        {"role": "assistant", "content": "Ask me to plan a trip for you."},
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

# --------------------------------------------------------------
# Step 1: Define the response format in a Pydantic model
# --------------------------------------------------------------

locally_stored_documents = ["auckland_attraction.pdf"]

class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""
    description: str = Field(description = "Raw description of the event")
    is_trip_planning_event: bool = Field(
        description = "Whether this text describes a trip planning event"
    )
    should_query_local_data: bool = Field(
        description = f"Whether this event involves travelling to one of the cities that has local data: {locally_stored_documents}"
    )
    trip_duration: str = Field(description = "Duration of the trip, if unknown, answer unknown")
    location: str = Field(description = "Location of the event")
    confidence_score: float = Field(description = "Confidence score between 0 and 1")

class Attraction(BaseModel):
    name: str = Field(description = "Name of the attraction")
    description: str = Field(description = "Description of the attraction")
    duration: str = Field(description = "Duration of the attraction")
    # confidence_score: float = Field(description = "Confidence score between 0 and 1")

class AttractionExtraction(BaseModel):
    attractions: list[Attraction] = Field(description = "List of attractions in this location")

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

# First LLM call to determine if input is a trip planning event
def analyse_input(user_input: str) -> EventExtraction:
    completion = client.beta.chat.completions.parse(
        model = "Gpt4o",
        messages = [
            {
                "role": "system",
                "content": f"Analyze if the text describes a trip planning event.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format = EventExtraction,
    )
    result = completion.choices[0].message.parsed
    return result

# Second LLM call to get attractions for a location (may or may not use local data)
def get_attractions(location: str, should_query_local_data: bool) -> AttractionExtraction:
    message = ''
    if should_query_local_data:
        local_data = query_local_data(location)
        message = f"Get attractions in this location: {location}, prioritise using the following local data: {local_data}."
    else:
        message = f"Get attractions in this location: {location}."
    completion = client.beta.chat.completions.parse(
        model = "Gpt4o",
        messages = [
            {
                "role": "system",
                "content": message,
            },
        ],
        response_format = AttractionExtraction,
    )
    result = completion.choices[0].message.parsed
    return result

# Third LLM call to generate itinerary
def generate_itinerary(attractions: list[Attraction], duration: str) -> str:
    message = ''
    if duration != 'unknown':
        message = f"Generate an itinerary for the following attractions: {attractions}, spread into {duration}."
    else:
        message = f"Generate an itinerary for the following attractions: {attractions}, spread into suitable days."
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

# Forth LLM call to generate itinerary
def get_trip_summary(user_input: str, itinerary: str) -> str:
    message = f"Using the itinerary: {itinerary}.\n\nSummarise a natural language response to the user's goal: {user_input}."
    # st.chat_message("system").write(message)
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

def call_function(name, args):
    if name == "analyse_input":
        return analyse_input(**args)
    elif name == "get_attractions":
        return get_attractions(**args)
    elif name == "generate_itinerary":
        return generate_itinerary(**args)
    elif name == "get_trip_summary":
        return get_trip_summary(**args)

def make_tool_calls(completion):
    st.session_state.messages.append(completion.choices[0].message)
    for tool_call in completion.choices[0].message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        if name == 'get_attractions' and args['should_query_local_data']:
            st.chat_message("assistant").write(f"Step: {name} for {args['location']} (use local data)")
        else:
            st.chat_message("assistant").write(f"Step: {name}")

        function_result = call_function(name, args)
        # st.chat_message("system").write(function_result)

        if type(function_result) == EventExtraction:
            function_result = function_result.model_dump()
        elif type(function_result) == AttractionExtraction:
            function_result = function_result.model_dump()
        
        st.session_state.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(function_result)})

def disable():
    st.session_state.disabled = True

prompt = st.chat_input(disabled = st.session_state.disabled, on_submit = disable)

if prompt != None and prompt != '':
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    sys_message = f"User wants to achieve the goal of {prompt}. Start with calling analyse_input, break it down into steps, in each step, only use tools provided."
    st.session_state.messages.append({"role": "system", "content": sys_message})
    completion = client.chat.completions.create(
        model = "Gpt4o",
        messages = st.session_state.messages,
        tools = tools,
    )
    result = completion.choices[0].message.content

    max_steps = 10
    step = 1
    
    while result == None and step <= max_steps:
        st.session_state.messages.append({"role": "system", "content": "What is the next step or next tool to call?"})
        # st.chat_message("system").write(st.session_state.messages)
        completion = client.chat.completions.create(
            model = "Gpt4o",
            messages = st.session_state.messages,
            tools = tools,
        )
        step += 1
        result = completion.choices[0].message.content
        if completion.choices[0].message.tool_calls != None:
            # st.chat_message("system").write(completion.choices[0].message.tool_calls)
            make_tool_calls(completion)
    
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)

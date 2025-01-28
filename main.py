import os
from openai import OpenAI

client = OpenAI(
    base_url = os.environ.get("OPENAI_BASE_URL"),
    api_key = os.environ.get("OPENAI_API_KEY")
)

print('\nSystem: Welcome! Let\'s start chatting! You may end a conversation by inputting \'End\'.')

conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."},
]

while True:
    end_conversation = False
    prompt = input('User: ')
    conversation_history.append(
        {"role": "user", "content": prompt}
    )

    if prompt.lower() == 'end':
        end_conversation = True
    
    try:
        chat_completion = client.chat.completions.create(
            messages=conversation_history,
            model="Gpt4o",
        )
        
        response = chat_completion.choices[0].message.content
        print(f"Assistant: {response}")
        conversation_history.append(
            {"role": "assistant", "content": response}
        )

        if end_conversation:
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

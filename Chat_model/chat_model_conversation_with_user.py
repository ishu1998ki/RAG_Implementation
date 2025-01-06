from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq

import os

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create a chat model
llm = ChatGroq(model="llama3-8b-8192")

chat_history = [] # Use a List to store the messages

# Set an initial system message
system_message = SystemMessage(content="You are a helpfull assistant.")
chat_history.append(system_message) # Add system message to chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) # Add user messages

    # Get AI response using history
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI messages

    print(f"AI: {response}")

print("___Message History___")
print(chat_history)
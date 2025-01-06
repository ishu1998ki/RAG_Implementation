import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192")

from langchain_core.messages import HumanMessage, SystemMessage

# messages = [
#     SystemMessage("Translate the following from English into Italian"),
#     HumanMessage("I love you!"),
# ]
messages = [
    SystemMessage("Devide the number user given by 2"),
    HumanMessage("8"),
]

results = llm.invoke(messages)

print(results)

print(results.content)
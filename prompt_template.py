from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create a chat model
model = ChatGroq(model="llama3-8b-8192")

# PART 1: Create a ChatPromptTemplate using a template string
print("-----Prompt from Template-----")
template = "Tell me a joke about {topic}."
promt_template =ChatPromptTemplate.from_template(template)

prompt = promt_template.invoke({"topic": "cats"})
results =model.invoke(prompt)

print(results.content)

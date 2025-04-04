from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from dotenv import load_dotenv

import os


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create a chat model
model = ChatGroq(model="llama3-8b-8192")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a advicor who tells advice about {topic}."),
        ("human", "Tell me {advice_count} advices."),
    ]
)
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"topic":"English speaking","advice_count":3})

# Output
print(result)
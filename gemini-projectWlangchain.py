import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=API_KEY,
    temperature=0.0,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | model

history = []

def chat(user_input: str) -> str:
    response = chain.invoke({"history": history, "input": user_input})
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))
    return response.content

print(chat("My name is Yazan, i am 19 yearsold."))
print(chat("What's my name?"))
print(chat("What's my age?"))
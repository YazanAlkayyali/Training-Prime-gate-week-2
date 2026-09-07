import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from .ai_config import API_KEY

Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)
langfuse_handler = CallbackHandler()

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
    response = chain.invoke({"history": history, "input": user_input}, config={"callbacks": [langfuse_handler]})
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))
    return response.content
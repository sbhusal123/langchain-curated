from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field

class ResponseFormatter(BaseModel):
    joke:str = Field(..., alias="Output Joke")
    topic: str=Field(..., alias="Topic of Joke")


chat_ollama = ChatOllama(model="llama3:latest").with_structured_output(ResponseFormatter)

res_chat_ollama = chat_ollama.invoke("Tell me a joke about chicken")

print(res_chat_ollama)

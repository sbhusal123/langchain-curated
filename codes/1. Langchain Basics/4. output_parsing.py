from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.output_parsers import (
    StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser, 
    ListOutputParser, PydanticOutputParser
)

chat_ollama = ChatOllama(model="llama3:latest")

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Tell me a joke about {topic}")
])

output = StrOutputParser()

chain = chat_prompt_template | chat_ollama | output

res = chain.invoke({"topic": "chicken"})

print(res)
# Output: Why did the chicken join a band? Because it had the drumsticks.

print(type(res))
# Output: <class 'str'>

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import OllamaLLM, ChatOllama

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

chat_ollama = ChatOllama(model='llama3:latest')

# with PromptTemplate
ollama = OllamaLLM(model='llama3:latest')
prompt_template = PromptTemplate.from_template('Translate following text to Spanish: {text}')
ollama_chain =  prompt_template | ollama # Lang Chain Chain => LCEL, here output from prompt_template is passed to ollama
prompt_resp = ollama_chain.invoke({'text': 'Hello, how are you?'})
print("LLM Resp",prompt_resp)

# with ChatPromptTemplate
chat_prompt_template = ChatPromptTemplate.from_template('Translate following text to Spanish: {text}')
chat_ollama_chain =  chat_prompt_template | chat_ollama # Lang Chain Chain => LCEL, here output from chat_prompt_template is passed to chat_ollama
chat_resp = chat_ollama_chain.invoke({'text': 'Hello, how are you?'})
print("Chat Resp",chat_resp)
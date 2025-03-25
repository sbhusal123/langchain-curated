from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

llm = OllamaLLM(model='llama3:latest')

# Chat Models Docs: https://python.langchain.com/docs/concepts/chat_models/
# Example From Tutorial:  https://python.langchain.com/docs/tutorials/llm_chain/#using-language-models
chat_llm = ChatOllama(model='llama3:latest')

if __name__ == "__main__":
    ip = input("Enter your prompt: ")

    # llm model
    # -----------------------------------------------------------
    # resp = llm.invoke(ip)
    # print(res)
    # print(type(res)) # <class 'str'>
    # -----------------------------------------------------------

    # chat model
    # --------------------------------------------------------------------------------------------------------
    res = chat_llm.invoke((
        SystemMessage(content="Your name is a ollama translator, your job is to translate to english"),
        HumanMessage("Bom dia?"),
        AIMessage("Bom dia translated to english is: heyy how are you doing?"),
        HumanMessage(ip),
    ))
    print(res)
    print(type(res)) # <class 'langchain_core.messages.ai.AIMessage'>
    # --------------------------------------------------------------------------------------------------------


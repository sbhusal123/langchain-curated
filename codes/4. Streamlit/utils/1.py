from langchain_ollama import ChatOllama

llm = ChatOllama(model='llama3:latest')
chat_history = []

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


message_history_context_system_prompt = """Given a chat history and the latest user question {input} \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

message_history_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", message_history_context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

document_qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question {input}. \
based on {context}.
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
"""
document_qa_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", document_qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

text = TextLoader('./faq.txt').load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(text)

embedings = OllamaEmbeddings(model='nomic-embed-text:latest')
db = Chroma.from_documents(docs, embedings)

retriver = db.as_retriever()

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# message history aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriver, message_history_chat_prompt)


# document aware chain
question_answer_chain = create_stuff_documents_chain(llm, document_qa_chat_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

query = "asd"

from langchain_core.messages import HumanMessage

def query_message(message):
    resp = rag_chain.invoke({
        "input": message,
        "chat_history": chat_history
    })    
    chat_history.extend([
        HumanMessage(content=query), resp["answer"]
    ])

if __name__ == "__main__":
    while True:
        query = input("Enter a query: ")
        resp = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        print("Response: ",resp)
        chat_history.extend([
            HumanMessage(content=query), resp["answer"]
        ])
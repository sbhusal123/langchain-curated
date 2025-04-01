from langchain_ollama import ChatOllama

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

def load_and_split_docs(document_path):
    """Loads and returns a DOcuments"""
    text = TextLoader(document_path).load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=150, chunk_overlap=0)
    docs = text_splitter.split_documents(text)
    return docs

def create_embedings(docs, model='nomic-embed-text:latest'):
    """Creates embeding and returns a retriever"""

    embedings = OllamaEmbeddings(model=model)
    db = Chroma.from_documents(docs, embedings)
    retriver = db.as_retriever()
    return retriver

message_history_context_system_prompt = """Given a chat history and the latest user question {input} \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


def create_message_history_aware_retriever(
        llm,
        retriver
):
    """Returns a retriever that is aware of chat history"""
    message_history_chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", message_history_context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriver, message_history_chat_prompt)

    return history_aware_retriever


document_qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question {input}. \
based on {context}.
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
"""


def create_docunent_qna_retriever(llm):
    document_qa_chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", document_qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(
        llm,
        document_qa_chat_prompt
    )
    return question_answer_chain



def create_history_aware_qna_rag_chain(
        model,
        embeding_model,
        document_path
):
    """Initializes and creates qna and history aware rag chain"""

    llm = ChatOllama(model=model)
    docs = load_and_split_docs(document_path)
    retriever = create_embedings(
        docs=docs,
        model=embeding_model
    )

    history_aware_retriever = create_message_history_aware_retriever(
        llm=llm,
        retriver=retriever
    )
    document_qna_retriver = create_docunent_qna_retriever(llm=llm)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_qna_retriver
    )

    return rag_chain

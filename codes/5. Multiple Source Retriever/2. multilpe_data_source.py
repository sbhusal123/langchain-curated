from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_chroma import Chroma

from pydantic import BaseModel, Field


from enum import Enum

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

class CategoryType(str, Enum):
    """Enumeration of  category types."""
    SHOES = 'SHOES'
    SHIRT ='SHIRT'

class Search(BaseModel):

    query: str = Field(..., description="Query to look up for.")

    category: CategoryType = Field(..., description="Category to look things up for")

llm = ChatOllama(model='llama3:latest')
llm_structured = llm.with_structured_output(Search)

system = """
You have the ability to issue search queries to get information to help answer user information. 
if you answer general inquiries, refer to the company name only without the clothes category (shirts, shoes ...).
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


# query analyzer analyzes person and query
query_analyzer = (
    {
        "question": RunnablePassthrough()
    }
    | prompt
    | llm_structured
)

# Prompt for retrival and generation:
retrival_system_prompt = """
You are a customer support specialist.

{question}

You assist with queries based on {context}
"""
retrival_prompt = ChatPromptTemplate.from_messages([
    ("system", retrival_system_prompt),
    ("human", "{question}")
])

embedings = OllamaEmbeddings(model='nomic-embed-text:latest')

# shirts retriver
raw_text_shirts = TextLoader('./docs/faq_shirts.txt').load()
splitter = CharacterTextSplitter(separator='.', chunk_size=100, chunk_overlap=0)
splits = splitter.split_documents(raw_text_shirts)
vector_store = Chroma.from_documents(documents=splits, embedding=embedings)
retriver_shirts = vector_store.as_retriever()

# shoes retriver
raw_text_shoes = TextLoader('./docs/faq_shoes.txt').load()
splitter = CharacterTextSplitter(separator='.', chunk_size=150, chunk_overlap=0)
splits = splitter.split_documents(raw_text_shoes)
vector_store = Chroma.from_documents(documents=splits, embedding=embedings)
retriver_shoes = vector_store.as_retriever()


from langchain_core.runnables import chain

retrivers = {
    "SHOES": retriver_shoes,
    "SHIRT": retriver_shirts
}

@chain
def custom_chain(question):
    resp = query_analyzer.invoke(question)

    retriver = retrivers[resp.category.value]
    
    chain = (
        {
            "context": retriver,
            "question": RunnablePassthrough()
        }
        | retrival_prompt
        | llm
        | StrOutputParser()
    )


    return chain.invoke(question)

resp = custom_chain.invoke("What is the return policy for shoes?")

print(resp)

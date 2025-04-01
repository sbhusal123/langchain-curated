from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_chroma import Chroma

from pydantic import BaseModel, Field

from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate

from enum import Enum
from pydantic import BaseModel

class PersonType(str, Enum):
    """Enumeration of allowed person types."""
    HARRISON = 'HARRISON'
    ANKUSH ='ANKUSH'


class Search(BaseModel):
    """Search for information about person."""
    query: str = Field(..., description='Query to look up.')
    person: PersonType = Field(..., description='Person to look things up for. Should be HARRISON or ANKUSH')



llm = ChatOllama(model='llama3:latest', temperature=0.7)
structured_llm = llm.with_structured_output(Search)

embedings = OllamaEmbeddings(model='nomic-embed-text:latest')

texts = ["Harrison worked at google."]
vector_store = Chroma.from_texts(texts=texts, embedding=embedings, collection_name='harrison')
retriver_harrison = vector_store.as_retriever()

texts = ["Ankush worked at facebook."]
vector_store = Chroma.from_texts(texts=texts, embedding=embedings, collection_name='ankush')
retriver_ankush = vector_store.as_retriever()

system = """
You have the ability to issue search queries to get information to help answer user information.
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


query_analyzer = (
    {
        "question": RunnablePassthrough()
    }
    | prompt
    | structured_llm
)


retrievers = {
    "HARRISON": retriver_harrison,
    "ANKUSH": retriver_ankush
}

# defining a custom tool for a chain
from langchain_core.runnables import chain

@chain
def custom_chain(question):
    structured_output = query_analyzer.invoke(question)
    retriver = retrievers[structured_output.person.value]
    return retriver.invoke(structured_output.query)

resp = custom_chain.invoke("Where did harrison worked")

print("resp", resp[0].page_content)


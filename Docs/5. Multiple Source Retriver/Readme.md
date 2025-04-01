# Dealing with Multiple Source and Multiple Retrivers

When we have multiple source text document containing a info about a service, product. We need to
incoperate it into out RAG.

We'll create a two different retrivers. It will automatically choose the proper retriver based on the call made to prompt to figure out a query and prompt. So, model will have a structured output and a tool to map the retriver to the invocation.


**Structured Output:**
```python
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
```

**Two of the retrivers:**

```python
embedings = OllamaEmbeddings(model='nomic-embed-text:latest')

texts = ["Harrison worked at google."]
vector_store = Chroma.from_texts(texts=texts, embedding=embedings, collection_name='harrison')
retriver_harrison = vector_store.as_retriever()

texts = ["Ankush worked at facebook."]
vector_store = Chroma.from_texts(texts=texts, embedding=embedings, collection_name='ankush')
retriver_ankush = vector_store.as_retriever()
```


**Query analysis and figuring out a question and person:**

```python
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

from langchain_core.runnables import chain

@chain
def custom_chain(question):
    structured_output = query_analyzer.invoke(question)
    retriver = retrievers[structured_output.person.value]
    return retriver.invoke(structured_output.query)

resp = custom_chain.invoke("Where did harrison worked")

print("resp", resp[0].page_content)
```

So, basically we've done a query analysis to figure out person and a query mapping, and then we selected a appropriate retriver and perform search.


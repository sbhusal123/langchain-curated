from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class FileOutput(BaseModel):
    filename: str = Field(description="The name of the file, must be text file")
    content: str = Field(description="Text content to be written into the file")

# Create the output parser
parser = PydanticOutputParser(pydantic_object=FileOutput)

template="""
Create a story avout topic: {topic} paragraph and a file name for it. 
Story must be a paragraph and a file name must be linked with a story
Use format below

{format_instructions}
"""

# Set up the prompt
prompt = PromptTemplate(
    template=template,
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Use an LLM (OpenAI in this case)
llm = ChatOllama(model='deepseek-r1:latest', temperature=0.5)

structured_llm = llm.with_structured_output(FileOutput)

chain = prompt | structured_llm

result = chain.invoke({"topic": "AI in Education"})

if __name__ == "__main__":
    inp = input("Enter a topic for the story: ")
    result = chain.invoke({"topic": inp})

    with open(result.filename, 'w') as file:
        file.write(result.content)
    print(f"File '{result.filename}' created with content.")

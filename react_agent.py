import os
from langchain_ollama import OllamaLLM
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.tools import Tool
import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup

# --- 1. Define the LLM (Ollama with DeepSeek) ---
ollama_llm = OllamaLLM(model='deepseek-r1:latest', temperature=0.7)
# If you have a different DeepSeek model name, replace 'deepseek-r1:latest'

# --- 2. Define the Tools ---
def search_internet(query: str) -> str:
    """Useful for when you need to answer questions about current events or general knowledge."""
    try:
        search_url = f"https://www.google.com/search?q={query}&num=2"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results = [result.get_text() for result in soup.find_all('div', class_='tF2Cxc')]
        return "\n".join(results[:2]) if results else "No relevant search results found."
    except Exception as e:
        return f"Error during search: {e}"

def get_current_datetime(location: str) -> str:
    """Useful for getting the current date and time for a specific location."""
    now = datetime.now()
    return f"The current date and time in {location} is: {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"

search_tool = Tool(
    name="internet_search",
    func=search_internet,
    description="Useful for when you need to answer questions about current events or general knowledge. Input should be a search query.",
)

datetime_tool = Tool(
    name="current_datetime",
    func=get_current_datetime,
    description="Useful for getting the current date and time for a specified location. Input should be the location.",
)

tools = [search_tool, datetime_tool]

prompt_string = """Answer the following questions as best you can. You have access to the following tools:

{tool_names_with_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

# --- 3. Define the Prompt Template for ReAct ---
prompt = PromptTemplate.from_template(
    template=prompt_string,
    partial_variables={
        "tool_names_with_descriptions": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        "tool_names": ", ".join([tool.name for tool in tools]),
        "tools": tools
    },
)

# --- 4. Create the ReAct Agent ---
agent = create_react_agent(llm=ollama_llm, tools=tools, prompt=prompt)

# --- 5. Create the Agent Executor ---
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 6. Run the Agent with a Question ---
question = "Show me current datetime."
print(f"Question: {question}")

response = agent_executor.invoke({"input": question})
print(f"Answer: {response['output']}")
print("\n\nResponse: \n\n\n", response)
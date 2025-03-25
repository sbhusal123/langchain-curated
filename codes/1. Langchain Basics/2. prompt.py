# prompt template: https://python.langchain.com/docs/tutorials/llm_chain/#prompt-templates
# Concept: https://python.langchain.com/docs/concepts/prompt_templates/

from langchain.prompts import PromptTemplate, ChatPromptTemplate

# String Prompt Template: https://python.langchain.com/docs/concepts/prompt_templates/#string-prompttemplates
prompt_template = PromptTemplate.from_template('What is a good name for a company that makes {product}?')
prompt_string = prompt_template.format(product='software')
print(prompt_string) # What is a good name for a company that makes software?

# ChatPromptTemplate: https://python.langchain.com/docs/concepts/prompt_templates/#chatprompttemplates
chat_prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])
chat_prompt = chat_prompt_template.format(topic='cats')
print(chat_prompt)
# System: You are a helpful assistant
# Human: Tell me a joke aboue cats
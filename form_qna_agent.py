from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from pydantic import BaseModel


class FormResponse(BaseModel):
    email: str
    password: str
    completed: bool
    message: str

system_prompt = """
You are an agent who helps in filling up a form.

Form has two fields to be filled:
- email
- password


Your response format will always JSON format like below:

{
   "email": "",
   "password": "",
   "completed": <true/false>,
   "message": ""
}

In above json schema:
- email is actual email (must be valid format email).
- password is actual password (must be at least 8 characters long and containing 1 special character and 1 number)
- completed represents weather email and password both are filled or not. (Boolean Field)
- message is a prompt message to be shown if any fields are missing or invalid.

Make sure email and password are valid and not a empty string.

Begin converstation by greeting user and asking to fill email. Message must always be one among:
- Hi, please enter your email.
- Hi, please enter your password.
- What's your email ?
- What's your password ?
- Ohh sorry that a wrong email format, please enter a valid email.
- Ohh sorry that a wrong password format, please enter a valid password.
- Great that's a valid email, please enter a password.
- Great that's a valid password, please enter a email.

Do not make up anything and always strictly respond in the above stated json format.
"""

message = [
    SystemMessage(content=system_prompt)
]

while x := input("Enter your message: "):
    if x.lower() == "exit":
        break
    message.append(HumanMessage(content=x))
    llm = ChatOllama(model="deepseek-r1:latest", temperature=0)

    structured_output_llm = llm.with_structured_output(FormResponse)

    output = structured_output_llm.invoke(message)
    if(output.completed):
        print("Response: ", output)
        break
    else:
        print("Response: ", output.message)
        message.append(AIMessage(content=str(output)))
        continue
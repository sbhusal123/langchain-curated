# Using Streamlit to build frontend for QNA Chatbot

We have a model chatbot interface accessed with function ``query_message``.

Session state can be accessed with: ``st.session_state``

Messages are stored in sesstion_state on chatbot initialization.

```python
st.session_state.messages = []
st.session_state.messages.append(
    {
        "role": "system",
        "content": "You are a helpful assistant."
    }
)
```

- ``prompt := st.chat_input("What is up ?")`` for input field with ``What's up?`` placeholder, input value in ``prompt``



```python
import streamlit as st

from utils.query import create_history_aware_qna_rag_chain


def query_message(query):
    chain = create_history_aware_qna_rag_chain(
        model='llama3:latest',
        embeding_model='nomic-embed-text:latest',
        document_path="./utils/faq.txt"
    )

    resp = chain.invoke({
        "input": query,
        "chat_history": st.session_state.get('messages', [])
    })

    return resp


st.title("Chatbot App")

chat_placeholder = st.empty()

def init_chat_history():
    """Initialize chat history with system message."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        )

def start_chat():
    """Start the chatbot converstation"""

    # show all the message with roles except for system
    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message['content'])
    
    # ask for input prompt in input
    if prompt := st.chat_input("What is up ?"):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # show user's message in UI.
        with st.chat_message("user"):
            st.markdown(prompt)

        # get response from model
        response = query_message(prompt)

        # show response answer in UI
        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        # update statw with response from model
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"]
        })

if __name__ == "__main__":
    init_chat_history()
    start_chat()

```
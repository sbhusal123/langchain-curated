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

    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message['content'])
    
    if prompt := st.chat_input("What is up ?"):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        response = query_message(prompt)

        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"]
        })

if __name__ == "__main__":
    init_chat_history()
    start_chat()

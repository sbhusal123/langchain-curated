import streamlit as st


st.title("Chatbot App")

chat_placeholder = st.empty()

def init_chat_history():
    """Initialize chat history with system message."""
    if "messages" in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append([
            {
                "role": "system",
                "content": "You are a helpful assistant. Ask me anything! "
            }
        ])

def start_chat():
    """Start the chatbot converstation"""
    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] == "system":
                with st.chat_message(message["role"]):
                    st.markdown(message['content'])
    
    if prompt := st.chat_input("What is up ?"):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.markdown("")

        st.session_state.messages.append({
            "role": "user",
            "message": prompt
        })

if __name__ == "__main__":
    init_chat_history()
    start_chat()

import streamlit as st
from rag_pipeline import app
from langchain_core.messages import HumanMessage
import uuid
import time
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Diabetes Chatbot")
st.caption("Ask me anything about Non-Communicable Diseases (NCDs) and Diabetes.")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle user input ---
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to session state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Pseudo-streaming assistant response ---
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        # Prepare input for LangGraph
        inputs = {"messages": [HumanMessage(content=prompt)]}
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Invoke the LangGraph app (synchronous)
        ans = app.invoke(inputs, config=config)
        # Extract the final answer text
        response_text = ans["messages"][-1].content
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        # Pseudo-streaming: display text character by character
        for char in response_text:
            full_response += char
            response_container.markdown(full_response + "▌")
            time.sleep(0.02)  # adjust speed for typing effect

        # Finalize the response (remove cursor)
        response_container.markdown(full_response)

    # Save assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})

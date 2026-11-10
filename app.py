
import asyncio

import streamlit as st

from pyqproj.vector_store import PolicyVectorStore
from pyqproj.rag_chain import HRPolicyRAG


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="centered"
)

st.title("📋 HR Policy Assistant")
st.caption("Powered by Gemini, LangChain and FAISS")


# Load the saved index only once per session
@st.cache_resource
def load_rag():

    store = PolicyVectorStore.load_local(
        "faiss_index"
    )

    rag = HRPolicyRAG(
        vector_store=store,
        hris_data_path="data/hris_api.json"
    )

    return rag


rag = load_rag()


# Sidebar for employee identification
with st.sidebar:

    st.header("Employee Settings")

    employee_id = st.text_input(
        "Employee ID",
        placeholder="Example: EMP101"
    )

    st.info(
        "Enter an employee ID to ask questions "
        "about personal leave balances."
    )


# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input(
    "Ask a question about HR policies..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching HR policies..."):

            try:
                response = asyncio.run(
                    rag.query(
                        question,
                        employee_id=employee_id or None
                    )
                )

                st.markdown(response)

            except Exception as e:
                response = f"An error occurred: {e}"
                st.error(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

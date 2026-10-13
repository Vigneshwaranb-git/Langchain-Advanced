import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Page Configuration
st.set_page_config(
    page_title="AI Chef & Nutritionist",
    page_icon="🍳",
    layout="centered"
)

st.title("🍳 AI Chef & Nutritionist")
st.write("Get 5 healthy, quick breakfast ideas using your favorite main ingredient!")

# 2. Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API Key not found! Please make sure it's set in your `.env` file or environment variables.")
    st.stop()

# 3. Initialize Model and Chain (Cached so it initializes only once)
@st.cache_resource
def get_conversational_chain():
    model = ChatOpenAI(model_name="gpt-4o-mini", api_key=api_key, temperature=0.0)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a professional nutritionist and expert chef. Provide exactly 5 healthy, quick breakfast ideas using the main ingredient provided by the user."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    base_chain = prompt_template | model | StrOutputParser()
    
    # In-memory store for session histories
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    conversational_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
    )
    return conversational_chain

conversational_chain = get_conversational_chain()

# 4. Initialize Streamlit Session State for UI Message History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Unique session config for LangChain
config = {"configurable": {"session_id": "streamlit_user_session"}}

# 5. Display Prior Chat Messages on App Re-run
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Capture User Input via Streamlit Chat Input Widget
if user_input := st.chat_input("Enter your main ingredient (e.g., oats, eggs, bananas)..."):
    # Append user message to UI state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in conversational_chain.stream({"input": user_input}, config=config):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"An error occurred: {e}"
            message_placeholder.markdown(full_response)
            
    # Append assistant response to UI state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
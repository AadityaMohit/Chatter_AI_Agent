import streamlit as st
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

# Fix for asyncio event loop issue on Windows
if sys.platform.startswith('win'):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Streamlit Page Configuration
st.set_page_config(page_title="AadityaMohit AI Chatbot", page_icon="🤖", layout="wide")

# Load Model and Tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"  # You can change this to "microsoft/DialoGPT-medium" for better responses
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize Chat History in Session State
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to Get Model Response
def get_response(user_input):
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Sidebar for User Input
with st.sidebar:
    st.title("💬 Chat with AI")
    user_input = st.text_area("You:", placeholder="Type your message here...", key="input", height=100)

    # Send Button
    if st.button("Send"):
        if user_input:
            # Add User Input to Chat History
            st.session_state.chat_history.append({"role": "user", "message": user_input})
            
            # Get Model Response with Error Handling
            try:
                response = get_response(user_input)
                st.session_state.chat_history.append({"role": "bot", "message": response})
            except Exception as e:
                st.session_state.chat_history.append({"role": "bot", "message": "Oops! Something went wrong. Please try again."})
            
            # Reset the Input Field using rerun
            st.rerun()  # Use rerun to reset the input field

# Display Chat History
st.title("🤖 AadityaMohit AI Chatbot")
st.markdown("Welcome to your personal AI assistant! Start chatting below.")
chat_container = st.container()

# Scrollable Chat History Display
with chat_container:
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['message']}")
        else:
            st.markdown(f"**AI:** {chat['message']}")

# Clear Chat History Button with Confirmation
if st.button("Clear Chat"):
    if st.confirm("Are you sure you want to clear the chat history?"):
        st.session_state.chat_history = []

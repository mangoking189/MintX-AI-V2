import streamlit as st
import openai
from openai import OpenAI
import time

# Configure the page
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize the OpenAI client
@st.cache_resource
def get_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
    )

client = get_client()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# Sidebar for configuration
with st.sidebar:
    st.title("🤖 AI Chatbot Settings")
    
    st.subheader("Model Configuration")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.0, 1.0, 0.8, 0.1)
    max_tokens = st.slider("Max Tokens", 512, 16384, 2048, 512)
    
    st.subheader("Chat Management")
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_started = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Current Model:** qwen/qwen3-coder-480b-a35b-instruct")
    st.markdown("**Provider:** NVIDIA API")

# Main chat interface
st.title("💬 AI Chatbot")
st.markdown("Chat with the powerful Qwen model powered by NVIDIA!")

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_started = True
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Generate response using NVIDIA API
            completion = client.chat.completions.create(
                model="qwen/qwen3-coder-480b-a35b-instruct",
                messages=st.session_state.messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Stream the response
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    full_response += chunk_content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.markdown(error_message)
            full_response = error_message
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Instructions for first-time users
if not st.session_state.chat_started:
    st.markdown("""
    ### Welcome to the AI Chatbot! 🎉
    
    **Get started by:**
    1. Typing a message in the chat input below
    2. Adjusting settings in the sidebar
    3. Starting a conversation!
    
    **Example questions you can ask:**
    - "Hello! Can you help me with coding?"
    - "Explain quantum computing in simple terms"
    - "Write a Python function to calculate fibonacci numbers"
    - "Help me debug this code: [your code here]"
    """)

# Footer
st.markdown("---")
st.markdown(
    "Powered by [NVIDIA AI Foundation Models](https://build.nvidia.com/) | "
    "Model: Qwen Coder 480B"
)
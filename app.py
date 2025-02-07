import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "Q and A Chatbot"
groq_api_key = os.getenv('GROQ_API_KEY')

# Check if API key exists
if not groq_api_key:
    st.error("Error: Missing GROQ API Key. Please check your .env file.")
    st.stop()

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful AI assistant. Please provide concise and accurate responses.'),
    ('user', 'Question: {question}')
])

def generate_response(question, llm, temperature, max_tokens):
    """Generate AI response using LangChain and Groq model"""
    llm = ChatGroq(model=llm, groq_api_key=groq_api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Sidebar - Model Selection & Parameters
st.sidebar.title("🔧 Settings")
llm = st.sidebar.selectbox("🤖 Select an AI Model", ['gemma2-9b-it', 'deepseek-r1-distill-llama-70b', 'llama-3.1-8b-instant'])
temperature = st.sidebar.slider("🎚️ Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
max_tokens = st.sidebar.number_input("📏 Max Tokens", min_value=50, max_value=500, value=150)

# Main UI
st.title("💬 AI-Powered Q&A Chatbot")
st.write("Ask me anything, and I'll provide the best possible answer!")

# Input field for user question
user_input = st.text_input("✍️ Your Question:", placeholder="Type your question here...")

# Process user input
if user_input:
    with st.spinner("Thinking... 🤖"):
        response = generate_response(user_input, llm, temperature, max_tokens)
    
    # Display response with collapsible expander
    with st.expander("🔍 Click to view answer"):
        st.write(response)
else:
    st.info("💡 Tip: Enter a question above to get started!")

# Footer
st.markdown(
    """
    ---
    💡 *Powered by Groq AI & LangChain | Created with ❤️ using Streamlit*  
    """, unsafe_allow_html=True
)

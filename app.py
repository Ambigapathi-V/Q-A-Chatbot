import streamlit as st
import groq
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import os
from dotenv import load_dotenv

load_dotenv() 

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Q&A Chatbot with OLLAMA'

# Prompt Template
Prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are the helpful assistant. Please respond to the user queries.'),
    ('user', 'Question: {question}')
])

def generate_response(question, engine, temperature, ):
    llm = Ollama(model=engine, temperature=temperature)
    output_parser = StrOutputParser()
    chain = Prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title of the app
st.title('🌟 Enhanced Q&A Chatbot with Groq 🌟')

## Sidebar for settings
st.sidebar.title('🛠️ Settings')

## Dropdown to select various Ollama models
llm = st.sidebar.selectbox(
    'Select a Ollama model', 
    ['llama3.2:1b', 
     'gemma2:2b']
)

## Adjust response parameters
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
max_tokens = st.sidebar.number_input('Max Tokens', min_value=50, max_value=300, value=150)

## Main Interface for user input
st.write('### 🤖 Ask me anything!')
user_input = st.text_input('You:', placeholder='Type your question here...')

if user_input:
    response = generate_response(user_input, llm, temperature)
    st.write('### 🤔 Response:')
    st.success(response)  # Display the response as a success message
else:
    st.warning('Please provide input')  # Prompt for user input

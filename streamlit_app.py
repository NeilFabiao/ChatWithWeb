import sys
__import__('pysqlite3')
import pysqlite3
sys.modules['sqlite3'] = sys.modules["pysqlite3"]

import os
import requests
from urllib.parse import urlparse
import shutil
import streamlit as st
from datetime import datetime, timedelta
import tiktoken
import sqlite3
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from openai import OpenAI

# Initialize session state for last activity
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.now()

# Function to update the session state variable to the current time
def update_activity():
    st.session_state.last_activity = datetime.now()

def check_activity():
    time_since_last_activity = datetime.now() - st.session_state.last_activity
    if time_since_last_activity > timedelta(minutes=10):  # 10 minutes of inactivity
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()  # Stop the Streamlit app
    elif time_since_last_activity > timedelta(minutes=2):  # More than 2 minutes of inactivity
        st.warning('You have been inactive for more than 10 minutes. The session will end after 2 more minute of inactivity.')

check_activity()  # Check for user activity at the start

llm_name = "gpt-3.5-turbo-1106"#gpt-4-0125-preview
llm = ChatOpenAI(model_name=llm_name, temperature=0.7)

def check_website(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False
        
def init_or_reset_vector_store(url):
    # Clear existing data if the URL has changed
    if 'last_url' not in st.session_state or url != st.session_state.last_url:
        # Resetting the vector store and related state
        st.session_state.vector_store = None  # This effectively resets the vector store
        st.session_state.chat_history = [AIMessage(content="Hello, I am Jarvis 🤖. How can I assist you today?")]  # Reset chat history too if URL changes
        st.session_state.last_url = url  # Update the last URL

    # Initialize a new vector store if it doesn't exist
    if st.session_state.vector_store is None:
        st.session_state.vector_store = get_vectorstore_from_url(url)  # Reinitialize vector store with the new URL

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=150)
    document_chunks = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    return vector_store

def get_combined_retriever_chain(vector_store, llm):
    retriever = vector_store.as_retriever()
    context_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Utilize the previous conversation with the user to guide the search.")
    ])
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a virtual assistant named Jarvis (🤖),\
        Utilize the context provided by the current website to inform your answers.  ---\:{context}\
        Today's date and time is {current_time}."), 
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    context_retriever_chain = create_history_aware_retriever(llm, retriever, context_prompt)
    conversational_rag_chain = create_stuff_documents_chain(llm, conversation_prompt)
    combined_retrieval_chain = create_retrieval_chain(context_retriever_chain, conversational_rag_chain)
    return combined_retrieval_chain

def get_response(user_input, current_time):
    combined_chain = get_combined_retriever_chain(st.session_state.vector_store, llm)
    response = combined_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,
        "current_time": current_time,
    })
    return response['answer']

st.set_page_config(page_title="Jarvis 🤖🔗 - Chat with websites", page_icon="🤖")
st.title("Jarvis 🤖🔗 - Chat with websites (Experimental stage - Beta)")

# Provide a short description of what the project is about along with a simple use case example
st.markdown("""
### Project Overview

Jarvis 🤖🔗 is designed to assist with summarization and question answering from the website [Lilian Weng's Blog Post](https://lilianweng.github.io/posts/2023-06-23-agent/) . 
Useful for providing answers to specific questions from extensive text materials, 
for anyone looking to quickly gather insights from web content. (For a full description, check the sidebar in the top right corner.)

#### Additional Use Cases:
- **Conversational AI for Business**: Corporate users can integrate Jarvis into their intranet to provide employees with instant answers from company documents, reducing the time spent searching for information.
- **Customer Support Automation**: Companies can use Jarvis to answer common customer queries based on their product manuals and FAQ sections, thereby improving the efficiency of customer service.

""", unsafe_allow_html=True)

# Long description of what the project is about along with a simple use case example
# Sidebar for description and use cases
with st.sidebar:
    st.markdown(""" ## Project Overview

This application, Jarvis 🤖🔗, is designed to assist with summarization and question answering from a specific website. It is particularly useful for extracting concise information and answering specific questions from extensive text materials, making it an ideal tool for researchers, students, and anyone looking to quickly gather insights from web content.

### Simple Use Case Example:
**Background**: Jordan, a student, needs to understand the latest trends in machine learning for a school project but has limited time to read through extensive materials.

**Use Case**: Jordan uses Jarvis 🤖🔗, which is set to analyze Lilian Weng's insightful blog post. Jordan asks, "Can you provide a summary of the main points discussed in this article?" Following the summary, Jordan queries more specific information, "What does Lilian Weng say about the future of reinforcement learning?"

**Outcome**: With Jarvis's assistance, Jordan quickly obtains a clear summary and specific answers, enhancing the efficiency of his research and enabling him to focus on compiling his project with well-informed content.

### Additional Use Cases:
- **Conversational AI for Business**: Corporate users integrate Jarvis into their intranet to provide employees with instant answers from company documents, reducing time spent searching for information.
- **Customer Support Automation**: Companies use Jarvis to answer common customer queries based on their product manuals and FAQ sections, improving customer service efficiency.

## Website for analysis:
[Lilian Weng's Blog Post](https://lilianweng.github.io/posts/2023-06-23-agent/)
""", unsafe_allow_html=True)

# Main area for chat or other interactive elements



# Continue with the rest of your app
# Set the default website URL and make it non-editable
website_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
st.text_input("Website URL", value=website_url, disabled=True, on_change=update_activity)

if website_url:
    if check_website(website_url):
        # Initialize or reset the vector store for the current URL
        init_or_reset_vector_store(website_url)

        user_query = st.chat_input("Type your message here...")
        if user_query:
            chat_history = st.session_state.chat_history
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = get_response(user_query, current_time)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            update_activity()

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    else:
        st.error("The website is not accessible or does not exist.")
else:
    st.info("Please enter a website URL above.")

check_activity()  # Check for user activity at the end


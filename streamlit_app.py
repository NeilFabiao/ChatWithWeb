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
from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from openai import OpenAI

# Clearing the session state by deleting each key-value pair
keys = list(st.session_state.keys())
for key in keys:
    del st.session_state[key]
st.cache_data.clear()
st.cache_resource.clear()
#st.rerun()  # Rerun the app, which now has an empty state

# Initialize session state for last activity
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.now()

# Function to update the session state variable to the current time
def update_activity():
    st.session_state.last_activity = datetime.now()

def check_activity():
    time_since_last_activity = datetime.now() - st.session_state.last_activity
    if time_since_last_activity > timedelta(minutes=3):  # 3 minutes of inactivity
        st.stop()  # Stop the Streamlit app
    elif time_since_last_activity > timedelta(minutes=2):  # More than 2 minutes of inactivity
        st.warning('You have been inactive for more than 3 minutes. The session will end after 1 more minute of inactivity.')

check_activity()  # Check for user activity at the start

llm_name = "gpt-3.5-turbo-0125"
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
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]  # Reset chat history too if URL changes
        st.session_state.last_url = url  # Update the last URL

    # Initialize a new vector store if it doesn't exist
    if st.session_state.vector_store is None:
        st.session_state.vector_store = get_vectorstore_from_url(url)  # Reinitialize vector store with the new URL

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=150)
    document_chunks = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    return vector_store

def get_combined_retriever_chain(vector_store, llm):
    retriever = vector_store.as_retriever()
    context_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Analyze the current website {context}. Utilize the previous conversation with the user \
        to guide the search. Focus solely on information pertinent to the current website, avoiding \
        unrelated details.---\:{chat_history}")
    ])
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a virtual assistant named Jarvis (🤖), designed to assist with learning.\
        Maintain a polite and engaging tone throughout the conversation.\
        Utilize the context provided by the current website to inform your answers. \
        If the answer is unclear, express uncertainty honestly without fabricating information. \
        Ensure responses are concise yet comprehensive as needed. Note: Today's date and time is {current_time}. ---\:{context}"), 
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
st.title("Jarvis 🤖🔗 - Chat with websites")

# Adjust the ratio or add more granularity if needed
col1, col2 = st.columns([0.8, 0.2], gap="small")

with col1:
    st.markdown("## Settings", unsafe_allow_html=True)  # Using markdown to closely match header style

with col2:
    st.write("")  # This can be used to adjust vertical alignment if necessary
    if st.button('Restart'):
        #st.session_state.clear()  # Clearing the session state
        #st.rerun()  # Rerun the app, which now has an empty state
        
        # Clearing the session state by deleting each key-value pair
        keys = list(st.session_state.keys())
        for key in keys:
            del st.session_state[key]
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()  # Rerun the app, which now has an empty state

# Continue with the rest of your app
website_url = st.text_input("Website URL", on_change=update_activity)


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


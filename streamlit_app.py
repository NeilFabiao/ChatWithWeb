import sys
__import__('pysqlite3')
import pysqlite3
sys.modules['sqlite3'] = sys.modules["pysqlite3"]

import os
import requests
from urllib.parse import urlparse
import shutil
import streamlit as st
from datetime import datetime
import tiktoken
import sqlite3
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.memory import ConversationBufferWindowMemory
#from langchain.chains import ConversationalRetrievalChain
#from langchain.prompts import PromptTemplate
from openai import OpenAI


llm_name = "gpt-3.5-turbo-0125"#"gpt-4-0125-preview";gpt-3.5-turbo-0125;gpt-4-0613
llm = ChatOpenAI(model_name=llm_name, temperature=0.7)

# Function to check website accessibility


#test using: https://lilianweng.github.io/posts/2023-06-23-agent/
def check_website(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

    
# Function to retrieve the url data and store into vectordatabase
def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=150)
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    #ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
    #DB_DIR: str = os.path.join(ABS_PATH, "db")

    # Define the base path and database directory
    ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
    DB_DIR: str = os.path.join(ABS_PATH, "chroma")  # Specify the directory path

    # Remove old database files if any
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "chroma")  # Specify the directory path
        
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(),persist_directory=DB_DIR)

    return vector_store
    

def get_combined_retriever_chain(vector_store, llm):
    # Define retrievers with different search types and prompts
    retriever = vector_store.as_retriever()

    # Prepare the context prompt
    context_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    # Prepare the conversation prompt including current date and time
    
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a virtual assistant named Jarvis (🤖), designed to help with learning. \
         Maintain a polite and engaging tone. Use the context from the website below to answer the question. \
         If unsure, say so without making things up. Keep answers concise but detailed when necessary.\
         Current date and time: {current_time}\
         ---\:{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # Prepare the current date and time
    date_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Current date and time: {current_time}"),
    ])

    # First create the retriever chain for fetching context
    context_retriever_chain = create_history_aware_retriever(llm, retriever, context_prompt)

    # Then create the document chain for answering based on retrieved documents
    conversational_rag_chain = create_stuff_documents_chain(llm, conversation_prompt)

    # Combine both into one retrieval chain
    combined_retrieval_chain = create_retrieval_chain(context_retriever_chain, conversational_rag_chain)

    return combined_retrieval_chain

def get_response(user_input,current_time):
    # Combine the creation and use of the context and conversational chains
    combined_chain = get_combined_retriever_chain(st.session_state.vector_store, llm)
    
    # Invoke the combined chain with chat history and user input
    response = combined_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,
        "current_time": current_time,
    })

    # Extract and return only the 'speak' part of the response
    #if 'thoughts' in response['answer'] and 'speak' in response['answer']['thoughts']:
    #    return response['answer']['thoughts']['speak']
    #else:
    #    # Return a default message or handle the absence of 'speak' however you prefer
    #    return response['answer']
    return response['answer']
        


    
# Streamlit app configuration
st.set_page_config(page_title="Jarvis 🤖🔗 - Chat with websites", page_icon="🤖")
st.title("Jarvis 🤖🔗 - Chat with websites")

# Main page settings
st.header("Settings")
website_url = st.text_input("Website URL")

if website_url != "":
    if check_website(website_url):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(website_url)

        user_query = st.chat_input("Type your message here...")
        if user_query:
            chat_history = st.session_state.chat_history
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = get_response(user_query,current_time)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

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

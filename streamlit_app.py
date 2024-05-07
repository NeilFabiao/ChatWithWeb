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
    # create a vectorstore from the chunks
    ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
    DB_DIR: str = os.path.join(ABS_PATH, "db")
    vector_store = Chroma.from_documents(document_chunks, embeddings,persist_directory=DB_DIR)
    return vector_store

def get_combined_retriever_chain(vector_store, llm):
    retriever = vector_store.as_retriever()
    context_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    #("user", "Utilize the previous conversation with the user to guide the search. ")
    
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a virtual assistant named Jarvis (🤖).\
        Answer the user's questions based on the below context:\n\n{context}. \
        Today's date and time is {current_time}."), 
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    '''
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a virtual assistant named Jarvis (🤖), designed to assist with learning.\
        Utilize the {context} provided by the current website (Lilian Weng's Blog Post) to inform your answers. \
        Today's date and time is {current_time}."), 
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    '''

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

    # Log the entire response for debugging
    print(current_time,"Received response:", response)  # or use logging.info() for production code
    
    # Check if the response is a dictionary with a 'thoughts' key
    if isinstance(response, dict) and 'thoughts' in response:
        thoughts_dict = response.get('thoughts', {})
        speak_text = thoughts_dict.get('speak', '')
        return speak_text
    # If the response is not a dictionary with 'thoughts' key, return the 'answer' part
    else:
        return response['answer']

st.set_page_config(page_title="Jarvis 🤖🔗 - (Experimental stage - Beta)", page_icon="🤖")
st.title("Jarvis 🤖🔗 - (Experimental stage - Beta)")

# Provide a short description of what the project is about along with a simple use case example
st.markdown("""
## Project Overview

Jarvis 🤖🔗 is designed to assist with summarization and question answering from the website [Lilian Weng's Blog Post](https://lilianweng.github.io/posts/2023-06-23-agent/) . 
Useful for providing answers to specific questions from extensive text materials, 
for anyone looking to quickly gather insights from web content. (For a full description, check the sidebar in the top right corner.)

#### Additional Use Cases:
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

# Set the default website URL and make it non-editable
website_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
st.text_input("Website URL", value=website_url, disabled=True)

# Check if a website URL has been provided.
if website_url:
    # Check if the provided website URL is accessible and valid.
    if check_website(website_url):
        # Initialize or reset the vector store for the current URL if the website is valid.
        init_or_reset_vector_store(website_url)

        # Prompt the user to type their message.
        user_query = st.chat_input("Type your message here...")
        # Check if the user has entered a query.
        if user_query:
            # Retrieve the current chat history.
            chat_history = st.session_state.chat_history
            # Get the current time in a specific format.
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Generate a response using the user's query and the current time.
            response = get_response(user_query, current_time)
            # Append the user's message to the chat history.
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            # Append the AI's response to the chat history.
            st.session_state.chat_history.append(AIMessage(content=response))

        # Display each message in the chat history.
        for message in st.session_state.chat_history:
            # Check if the message is from the AI and display it appropriately.
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            # Check if the message is from the Human and display it appropriately.
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    else:
        # Display an error if the website is not accessible or does not exist.
        st.error("The website is not accessible or does not exist.")
else:
    # Prompt the user to enter a website URL if none is provided.
    st.info("Please enter a website URL above.")


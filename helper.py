import json
import os
from dotenv import load_dotenv, dotenv_values

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import secrets
import string
import asyncio

load_dotenv()

def generate_unique_id(length):
    characters = string.ascii_letters + string.digits  # set of charater to take from: A-Z, a-z, 0-9
    return ''.join(secrets.choice(characters) for _ in range(length))

def history_manager(chat_history, message, max_no_history_message: int = 8) -> None:

    """        
        Adding chat-history to the chat-history object
        Handle the maximun amount of chat history available
        
        Agrs: 
            Chat_history: array storing user's chat message
            message_user: user's message      (Format: ("human", question))
            message_ai: bot's message         (Format: ("ai", answer))
            max_no_history_message: max number of chat allowed in the history array
            
        Returns: 
            None
    """
    
    if len(chat_history) <= max_no_history_message:
        chat_history.append(message)
    else: 
        # Remove the oldest pair of messages
        chat_history.pop(0)
        chat_history.pop(0)
        # Add the new message
        chat_history.append(message)

def json_parser(output):
    return json.loads(output)

def load_and_preprocess_documents(folder_path: str):
    """
        Load the docmumet from local and prepare it for vector database
        
        Args:
            file_path (str): the path of the file
            
        Returns: 
            List[Docs]: the processed document
    """
    
    docs = []
    
    # List all the files in the folder
    file_list = os.listdir(folder_path)
    
    for file in file_list:
        file_location = f"{folder_path}/{file}"
        
        loader = PyPDFLoader(file_location)  # Load PDF
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc = text_splitter.split_documents(documents)
        
        docs.extend(text_splitter.split_documents(documents))
    
    return docs

def load_and_preprocess_website(url: str):
    """
        Scrape the website using the given url and prepare it for vector database
        
        Args:
            url (str): the website URL
            
        Returns: 
            List[Docs]: the processed document
    """    
    
    loader = WebBaseLoader(url)  # Load PDF
    documents = loader.load()
    
    print(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)  # Split into chunks
    return docs

def create_and_save_faiss_index(folder_path, index_path="vector_store"):
    """
        Create the FAISS index and save it to local
        (This function meant to create the FAISS index when you 
        first start the program and FAISS index doesn't exsit)
        
        Args: 
            folder_path (str): path storing the documents
            index_path (str): path storing the index
            
        Returns: 
            None
        
        Todo: 
            Need to create from multiple files
    """
    # Load and preprocess all documents inside the "folder_path" path
    documents = load_and_preprocess_documents(folder_path)
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    
    vector_store = FAISS.from_documents(documents, embeddings)  # Create FAISS index

    # Save FAISS index to disk
    vector_store.save_local(index_path)
    print(f"FAISS index saved at: {index_path}")
   
if __name__ == "__main__":
    create_and_save_faiss_index("../uploads")
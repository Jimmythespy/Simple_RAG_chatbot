from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from helper import create_and_save_faiss_index

import os

# Vector Store ---------------------------------------------------------------------------------
# Initialize embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large", 
)

vector_store = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)

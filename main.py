from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import os
from dotenv import load_dotenv
import aiofiles
import asyncio
import pprint

from helper import (
    load_and_preprocess_documents, 
    load_and_preprocess_website, 
    history_manager,
    generate_unique_id
)

from chain import suggestion_chain, llm, reformat_chain
from vector_db import vector_store
from prompt_template import prompt

load_dotenv()       # Load OpenAI API key

# Global variable -------------------------------------------------------------------------------
app = FastAPI() # Initialize FastAPI app
templates = Jinja2Templates(directory="templates") # Set up Jinja2 template directory

# Chat history
max_no_history_message = 5
chat_history = []  

# Query request
requests = {}

async def llm_stream(query):
    prompt_value = prompt.invoke(query)
    
    async for chunk in llm.astream(prompt_value):
        yield f"data: {chunk}\n\n"  # SSE format
        # print(chunk)
        await asyncio.sleep(0)      # Allow event loop to run other tasks
    
    print("Done")

# FastAPI-----------------------------------------------------------------------------------------
# FastAPI Query stucture
class Query(BaseModel):
    question: str
    time: int | None = None
    stream_option: bool | None = None
    
class Url_Upload(BaseModel):
    url: str

@app.get("/query_ui", response_class=HTMLResponse)
async def query_form(request: Request):
    
    """
        Endpoint to serve query UI
        \n When the query_ui is served, chat-history and requests will be cleared
    """
    
    global chat_history
    chat_history = []
    requests = {}
    return templates.TemplateResponse("query.html", {"request": request})

@app.get("/uploadfile_ui", response_class=HTMLResponse)
async def upload_form(request: Request):
    
    """
        Endpoint to serve HTML page for file upload
    """
    
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/listfile")
async def list_file(): 
    
    """
        Endpoint to return the list of file in the vector store
    """
    
    file = os.listdir("uploads/")
    return {"file_list" : file}

@app.post("/uploadfile")
async def upload_file(file: UploadFile = File(...)):
    
    """
        Handle file upload
    """
    
    global vector_store
    global embedding_model
    
    file_location = f"uploads/{file.filename}"
    
    # Write file to local
    async with aiofiles.open(file_location, 'wb') as f:
        content = await file.read()  
        await f.write(content)       
    
    # Load the newly downloaded document
    docs = load_and_preprocess_documents(file_location) 
    
    try:
        # Add document to vector store and save the index to local
        await vector_store.aadd_documents(docs)
        vector_store.save_local("vector_store")
        
    except Exception as e: 
        print(e)
        raise HTTPException(status_code=500, detail=str(e)) 
    
    return {"filename": file_location}

@app.post("/loadweb")
async def load_url(url: Url_Upload):
    
    """
        Handle website scrapping and store in database
    """
    
    global vector_store
    
    docs = load_and_preprocess_website(url.url) 
    
    try:
        await vector_store.aadd_documents(docs)
        vector_store.save_local("vector_store")
        
    except Exception as e: 
        print(e)
        raise HTTPException(status_code=500, detail=str(e)) 
    
    return {"url": url}

@app.get("/stream")
async def stream_response(request_id: str):
    """
        Handle SSE streaming to the client
        First need to request streaming to get streaming ID first
    """
    
    global vector_store
    global chat_history
    global requests
    
    optimized_answer = reformat_chain.invoke({
        "chat_history" : chat_history,
        "input" : requests[request_id]
    })
    
    print("optimized_answer", optimized_answer.content)
    
    # Search for context
    context = vector_store.similarity_search_with_relevance_scores(optimized_answer.content, k=20, score_threshold=0.3)
    context_combined = []
    
    print("context len: ", len(context))
    
    if len(context) == 0:
        context_combined = "None"
    else:
        for ct in context:
            print(ct[0].page_content)
            context_combined.append(ct[0].page_content)
    
    # Save the source of the context (This only take the 1st context)
    requests[request_id]["source"] = "" if (len(context) == 0) else context[0][0].metadata
    
    history_manager(chat_history, ("human", requests[request_id]["question"]))
    
    return StreamingResponse(
        llm_stream({
            "chat_history" : chat_history, 
            "context" : context_combined, 
            "input" : optimized_answer.content}), 
        media_type="text/event-stream"
    )
    
@app.post("/start-stream")
async def start_stream(query: Query):
    
    """
        Endpoint request to start stream, 
        sever generate an ID then send this ID to the client
    """
    
    global requests
    
    request_id = generate_unique_id(8)  # Generate a simple unique ID based on prompt length
    requests[request_id] = {
        "question" : query.question
    }
    
    # print("start_stream", requests)
    return {"request_id": request_id}

@app.get("/source")
async def stream_response(request_id: str):
    global requests
    # print(requests[request_id])
    
    try:
        return {"source" : requests[request_id]["source"]}
    except: 
        return {"source" : ""}

# API endpoint for streaming LLM responses
@app.post("/suggestion")
async def stream_response(query: Query):
    global suggestion_chain
    global vector_store
    
    related = False
    
    # Search for context
    context = vector_store.similarity_search_with_relevance_scores(query.question, k=2, score_threshold=0.3)
    
    if len(context) != 0:
        related = True
    
    result = suggestion_chain.invoke({"input" : query.question, "related" : related})
    print("result", result)
    return {result.content}
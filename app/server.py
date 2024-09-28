import calendar
from concurrent.futures import ThreadPoolExecutor
import gzip

import multiprocessing
from multiprocessing import Manager

import re
import signal
import ssl
import subprocess
import sys
import threading
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from bs4 import BeautifulSoup
from xvfbwrapper import Xvfb
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

import asyncio
from typing import List, Generator, Union, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import time
import uuid
import secrets
import nbformat
import io
import os
import random
import shutil
import json
import httpx
from http.server import BaseHTTPRequestHandler
import requests
import http.server
import socketserver
import nest_asyncio
nest_asyncio.apply()

from datetime import datetime
from threading import Thread, Event

from fastapi import FastAPI, File, Header, UploadFile, Form, HTTPException, Request, Depends, Cookie, APIRouter, BackgroundTasks
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from sse_starlette import EventSourceResponse
except ImportError:
    EventSourceResponse = Any

from pydantic import BaseModel, EmailStr, ValidationError

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Boolean, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import smtplib

import urllib
from urllib.parse import parse_qs

from nbclient import NotebookClient

from pathlib import Path

from langserve import add_routes

from starlette.staticfiles import StaticFiles
from starlette.responses import Response #, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.datastructures import Headers

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import Language
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents.base import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError

from importer.load_and_process import FileEmbedder
from app.config import EMBEDDING_MODEL
from app.embeddings_manager import get_embedding_from_manager, get_session_vector_store, reset_embeddings_storage, reset_session_vector_store, set_current_session_id, store_embedding, PGVectorWithEmbeddings, store_driver, store_window_handler, reset_driver, reset_window_handlers
from app.rag_chain import final_chain, random_base36_string, copy_driver_session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/rag/static", StaticFiles(directory="./source_docs"), name="static")

### Escalator Shit
PORT = 3001

proxy_proc = None
global_cookies_dict = {}
# Initialize manager and shared list
manager = Manager()
global_cookies_list = manager.list()

def close_all_windows(driver: uc.Chrome):
    """
    Close all windows in the driver except one and navigate it to about:blank.
    """
    # Get all window handles
    window_handles = driver.window_handles
    
    # Close all windows except the last one
    for handle in window_handles[:-1]:
        driver.switch_to.window(handle)
        driver.close()

    # Switch to the remaining window and navigate to about:blank
    driver.switch_to.window(driver.window_handles[0])
    driver.get("about:blank")
    
# Serve service-worker-popup.js at /service-worker-popup.js
@app.get("/service-worker-popup.js", response_class=FileResponse)
async def service_worker_js():
    # file_path = os.path.join("../frontend/public/service-worker-popup.js")
    file_path = './app/service-worker-popup.js'
    return FileResponse(file_path, media_type="application/javascript")

# executor = ThreadPoolExecutor(max_workers=1)
# @app.post("/start-networking")
# async def start_networking(request: Request):
#     request_json = await request.json()
#     print(f"start networking request json: {request_json}")
#     job_id = request_json['input']['id']
#     type = request_json['input']['type']
    
#     job_posting_embedding, profile_data = get_job_data(job_id)
    
#     initial_inputs = [
#         {
#             "instruction": "Extract relevant profile keywords given your context",
#             "url": profile.get("url"),
#             "image_path": profile.get("image_path"),
#             "name": profile.get("name"),
#             "id": random_base36_string(),
#             "type": type,
#             "session_id": job_id
#         }
#         for profile in profile_data[1::2]
#     ]
    
#     async def create_task(input_data):
#         return await profile_preprocessing_chain.ainvoke(input_data)
    
#     def chunk_list(lst, n):
#         for i in range(0, len(lst), n):
#             yield lst[i:i + n]
    
#     num_chunks = 2
#     chunk_size = (len(initial_inputs) + num_chunks - 1) // num_chunks
#     input_chunks = list(chunk_list(initial_inputs, chunk_size))
    
#     async def event_generator():
#         preprocessing_results = []
#         retrieved_profiles = []
        
#         try:
#             for input_chunk in input_chunks:
#                 tasks = [create_task(input_data) for input_data in input_chunk]
#                 semi_global_driver = copy_driver_session(escalator_global.scraper.driver)
#                 close_all_windows(semi_global_driver)
#                 await asyncio.sleep(1)
                
#                 urls = []
#                 counter = 0
#                 for i, input_data in enumerate(input_chunk):
#                     if counter % 4 == 0:
#                         await asyncio.sleep(3)
#                     semi_global_driver.execute_cdp_cmd("Target.createTarget", {"url": input_data['url'], "newWindow": False})
#                     urls.append(input_data['url'])
#                     await asyncio.sleep(2)
#                     store_window_handler(unique_id=input_data['url'], window_handler=semi_global_driver.window_handles[i + 1])
#                     counter += 1

#                 await asyncio.sleep(3)
#                 store_driver(unique_id=job_id, driver=semi_global_driver)
                
#                 # chunk_results = await asyncio.gather(*tasks)
#                 # Run await asyncio.gather(*tasks) in a separate thread
#                 loop = asyncio.get_event_loop()
#                 chunk_results = await loop.run_in_executor(executor, lambda: asyncio.run(asyncio.gather(*tasks)))
                
#                 semi_global_driver.quit()
#                 reset_driver()
#                 reset_window_handlers()
                
#                 preprocessing_results.extend(chunk_results)
                
#             retrieved_profiles = retrieve_top20_profiles(session_id=job_id, job_posting_embeddings=job_posting_embedding, preprocessing_results=preprocessing_results)
#             unique_profiles_dict = {list(profile.values())[0]: profile for profile in retrieved_profiles}
#             profile_data_file = f"./database/profiles_{job_id}.json"
#             with open(profile_data_file, "w", encoding="utf-8") as f:
#                 json.dump(unique_profiles_dict, f)
            
#             yield json.dumps({"event": "data", "id": job_id, "status": "completed", "profiles": retrieved_profiles})
            
#         except Exception as e:
#             raise f"An error occured while networking for {job_id}: {e}"
        
#         finally:
#             # print(f"finally filtered: {retrieved_profiles}")
#             pass

#     return EventSourceResponse(event_generator(), media_type="text/event-stream") # media_type="text/event-stream"
###

# Mapping from file extensions to Language enumeration values
language_map = {
    'py': Language.PYTHON,
    'js': Language.JS,
    'ts': Language.TS,
    'html': Language.HTML,
}

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Upload file endpoint
@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...), session_id: str = Header(default_factory=secrets.token_urlsafe)):
    set_current_session_id(session_id)  # Update the global session ID
    upload_folder = Path("./uploaded_files")
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize the embeddings and text splitter
    embedding_method = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), show_progress_bar=True, skip_empty=True, chunk_size=100)
    text_splitter = SemanticChunker(embeddings=embedding_method)
    
    # Use session_id to handle embeddings in a session-specific manner.
    file_embedder = FileEmbedder(session_id)
    
    file_responses = []  # Initialize an empty list to store responses
    for file in files:
        unique_id = str(uuid.uuid4())
        save_path = upload_folder / f"{file.filename}"

        with open(save_path, "wb") as out_file:
            out_file.write(await file.read())
            
            # Determine file type and set parameters for processing
            file_extension = file.filename.split('.')[-1].lower()
            print(f"uploaded file_extension: {file_extension}")
            try:
                if file_extension in ['pdf', 'ipynb', 'txt', 'csv', 'png']:
                    glob_pattern = f"**/*.{file_extension}"
                    await file_embedder.process_and_embed_file(unique_id=unique_id, directory_path=upload_folder, file_path=save_path, embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter)
                elif file_extension in ['py', 'js', 'ts', 'html']:
                    suffixes = ["."+file_extension] # [".py"]
                    print(suffixes)
                    await file_embedder.process_and_embed_file(unique_id=unique_id, directory_path=str(upload_folder), file_path=save_path, embeddings=embedding_method, suffixes=suffixes, language_setting=language_map.get(file_extension, None))
                file_responses.append({"filename": file.filename, "unique_id": unique_id})
            except Exception as e:
                print(f"Error processing file {file.filename}: {e}")
            finally:
                # Delete the file after processing it to avoid redundancy
                save_path.unlink(missing_ok=True)  # Use missing_ok=True to ignore the error if the file doesn't exist
                if file_extension == "png":
                    image_txt_path = os.path.splitext(os.path.basename(file.filename))[0] + ".txt"
                    image_txt_save_path = upload_folder / f"{image_txt_path}"
                    image_txt_save_path.unlink(missing_ok=True)
            # print(f"file added to: {file_responses}")
            
    if not file_responses:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    
    return {"files": file_responses}

@app.get("/embeddings/{unique_id}")
async def get_embedding(unique_id: str):
    # Retrieve the stored embedding
    embedding = get_embedding_from_manager(unique_id)
    if embedding is None:
        print("Why is fetched embedding from current embedding_store none?")
        raise HTTPException(status_code=404, detail="Embedding not found")
    return embedding

@app.get("/files/{file_name}")
async def get_file(file_name: str):
    file_path = f"./uploaded_files/{file_name}"
    return FileResponse(path=file_path, filename=file_name, media_type='application/octet-stream')

@app.post("/api/submit-url")
async def submit_url(url: str = Form(...)):
    return JSONResponse(content={"url": url})

add_routes(app, final_chain, path="/rag") # Main

# add_routes(app, preprocess_chain, path="/preprocess")

# add_routes(app, research_chain, path="/research")

# add_routes(app, web_scraping_chain, path="/scrape")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

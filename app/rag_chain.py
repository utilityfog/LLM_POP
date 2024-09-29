from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import random
import re
import os
import string
import traceback
from urllib.parse import urljoin, urlparse
import uuid
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import undetected_chromedriver as uc
import aiofiles
import asyncio
import time

from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
from abc import ABC
from functools import partial
from dotenv import load_dotenv

from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
    LongContextReorder
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_core.vectorstores import VectorStoreRetriever, Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import Runnable
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.vectorstores.deeplake import DeepLake

from langchain.retrievers import (
    ContextualCompressionRetriever, 
    EnsembleRetriever, 
    MergerRetriever
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from pypdf import PdfReader

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from selenium import webdriver

from pyvirtualdisplay import Display

from fastapi import FastAPI, File, Header, UploadFile, Form, HTTPException, Request, Depends, Cookie, APIRouter, BackgroundTasks

try:
    from sse_starlette import EventSourceResponse
except ImportError:
    EventSourceResponse = Any
    
from pydantic import BaseModel

from importer.load_and_process import chatgpt_base, FileEmbedder
from app.embeddings_manager import get_current_session_id, get_embedding_from_manager, get_embeddings_storage, get_persistent_vector_store, get_session_vector_store, EMBEDDINGS_STORAGE, reset_embeddings_storage, reset_session_vector_store, set_current_session_id, PGVectorWithEmbeddings
from app.config import PG_COLLECTION_NAME, EMBEDDING_MODEL
from app.Automation import Automator
from app.ExtractorClasses import Extractor

load_dotenv()

# Google Search Environmental Variables
google_api_key = os.environ.get('GOOGLE_API_KEY')
google_cse_id = os.environ.get('GOOGLE_CUSTOM_ENGINE_ID')

def augment_context_with_file_embeddings(context: str, file_embedding_keys: Optional[Optional[str]]) -> str:
    augment_text = ""
    if file_embedding_keys:
        print(f"file_embedding_keys current: {file_embedding_keys}")
        for file_embedding_key in file_embedding_keys:
            file_embedding = get_embedding_from_manager(file_embedding_key)
            if file_embedding:
                augment_text += f"\nFrom {file_embedding_key}: {file_embedding}"
            else:
                print("Why is fetched embedding from current embedding_store none?")
        augment_text = f"\n augment: {augment_text}, Context: " + f"{context}"
        # trimmed_print("final augmented context", augment_text, 10000) # This does print something
    reset_embeddings_storage() # Reset after each augmented generation
    reset_session_vector_store(get_current_session_id()) # Reset after each augmented generation
    set_current_session_id("") # Reset after each augmented generation regardless of whether or not user has uploaded files
    return augment_text

# Initialize the embeddings method
embedding_method = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), show_progress_bar=True, chunk_size=100, skip_empty=True)

persistent_vector_store = get_persistent_vector_store()

# PGVector(
#     collection_name=PG_COLLECTION_NAME,
#     connection_string=os.getenv("POSTGRES_URL"),
#     embedding_function=embedding_method
# )

# Initialize DeepLake dataset
# deeplake_path = os.path.expanduser("~/deeplake_data")
# deeplake_db = DeepLake(dataset_path=deeplake_path, embedding_function=embeddings, read_only=True)

# initialize the ensemble retriever
# ensemble_retriever = EnsembleRetriever(retrievers=[vector_store.as_retriever(), deeplake_db.as_retriever()], weights=[0.5, 0.5])

# Default PGVectorStore retriever
pgvector_default_retriever = persistent_vector_store.as_retriever(
    search_type="mmr",
    # search_kwargs={"k": 6, "include_metadata": True}
    search_kwargs={'k': 6, 'lambda_mult': 0}
)
# With Score PGVectorStore retriever
pgvector_score_retriever = persistent_vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.01}
)

# deeplake_retriever = deeplake_db.as_retriever(
#     search_type="mmr", search_kwargs={"k": 9, "include_metadata": True}
# )

# The Lord of the Retrievers will hold the output of both retrievers and can be used as any other
# retriever on different types of chains.
lotr = MergerRetriever(retrievers=[pgvector_default_retriever])

# This filter will divide the documents vectors into clusters or "centers" of meaning.
# Then it will pick the closest document to that center for the final results.
# By default the result document will be ordered/grouped by clusters.
filter_ordered_cluster = EmbeddingsClusteringFilter(
    embeddings=embedding_method,
    num_clusters=3,
    num_closest=1,
)

# If you want the final document to be ordered by the original retriever scores
# you need to add the "sorted" parameter.
filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=embedding_method,
    num_clusters=2,
    num_closest=1,
    # random_state=None,
    sorted=True
    # remove_duplicates=True
)

# We can remove redundant results from both retrievers using yet another embedding.
# Using multiples embeddings in diff steps could help reduce biases.
embeddings_filter = EmbeddingsRedundantFilter(embeddings=embedding_method, similarity_threshold=0.97)
pipeline = DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever, embeddings_filter])
# Default
ordered_compression_default_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=pgvector_default_retriever
)
# Score retriever
ordered_compression_score_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=pgvector_score_retriever
)

### TEMPLATES
CONVERSATION_TEMPLATE = """
Answer given the following:

Uploaded file augment: 
{augment}

Retrieved context: 
{context}

Question from: 
{question}
"""

ARGUMENT_SCRAPING_TEMPLATE = """
Extract relevant Tourpedia API arguments from the input user prompt given the following:

Detailed Prompt:
{prompt}

Documentized Prompt Text:
{context}

File Embedding Keys:
{file_embedding_keys}

ID:
{id}

Return the extracted API arguments as parameters to a function call
"""

LOCATION_GENERATION_TEMPLATE = """
Extract relevant location properties from the given location review given the following:

Detailed Prompt: 
{instruction}

Vector Embedded Location Review: 
{context}

Return the extracted properties as arguments to a function call
"""

LOCATION_GUIDE_TEMPLATE = """
Provide a friendly guide of locations a user would like to visit based on the given location reviews corresponding to the prompt, given the following:

Detailed Prompt: 
{prompt}

Vector Embedded Location Reviews for Locations related to the prompt: 
{context}
"""
###

def template_processor(embeddings_storage, template):
    final_template = template
    # for key in embeddings_storage:
    #     print(f"appending key {key} to template")
    #     final_template += f"""\nInformation from file with {key}"""
    print(final_template)
    # print(traceback.print_stack())
    return final_template

# Refer to the uploaded file information: {file_embedding_key}
#ANSWER_PROMPT = ChatPromptTemplate.from_template(template_processor(embeddings_storage, template))

### LLMs
llm_conversation = chatgpt_base
llm_tourpedia_api_arguments_extractor = ChatOpenAI(temperature=0.3, model='gpt-3.5-turbo-0125', max_tokens=4096, streaming=True, tools=Extractor.extract_relevant_api_arguments_json, tool_choice={"type": "function", "function": {"name": "extract_relevant_api_arguments"}})
llm_tourpedia_location_object_generator = ChatOpenAI(temperature=0.3, model='gpt-3.5-turbo-0125', max_tokens=4096, streaming=True, tools=Extractor.extract_relevant_location_properties_json, tool_choice={"type": "function", "function": {"name": "extract_relevant_location_properties"}})
###

print(f"max_tokens: {llm_conversation.max_tokens}")
print(f"model_name: {llm_conversation.model_name}")

class RagInput(TypedDict, total=False):
    question: str
    fetchContext: Optional[bool] # Optional bool for whether or not to fetch context
    file_embedding_keys: Optional[List[str]] # Optional list of strings
    
class JupyterRagInput(TypedDict, total=False):
    question: str

# For Direct String output
# {
#     "context": (itemgetter("question") | vector_store.as_retriever()),
#     "question": itemgetter("question")
# } | ANSWER_PROMPT | llm | StrOutputParser()

# class print_stack_class:
#     def print_stack(self, **kwargs):
#         print(traceback.print_stack())
        
# print_stack_class_instance = print_stack_class()

# HELPERS
def trimmed_print(label, data, max_length=50):
    data_str = str(data)
    trimmed_data = (data_str[:max_length] + '...') if len(data_str) > max_length else data_str
    print(f"{label}: {trimmed_data}")
    
def random_base36_string(length=13):
    # Generate a random integer
    random_number = random.randint(0, 36**length - 1)
    
    # Convert the random number to a base 36 string
    base36_string = ""
    characters = string.digits + string.ascii_lowercase
    
    while random_number > 0:
        random_number, remainder = divmod(random_number, 36)
        base36_string = characters[remainder] + base36_string
    
    # Ensure the string is of the specified length
    return base36_string.zfill(length)

# Function to extract the filename from Content-Disposition header or URL
def get_filename_from_response(response, url):
    # Try to get filename from Content-Disposition header
    if 'Content-Disposition' in response.headers:
        content_disposition = response.headers['Content-Disposition']
        filename = None
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[-1].strip(' "')
        elif 'filename*=' in content_disposition:
            filename = content_disposition.split('filename*=')[-1].strip(' "')
        if filename:
            return filename
    # Fallback to extracting the filename from the URL
    parsed_url = urlparse(url)
    return parsed_url.path.split('/')[-1]

# PARTIALS
def get_augment_and_trace(dictionary):
    """Method to augment response with uploaded file embeddings"""
    keys = dictionary['file_embedding_keys']
    session_id = get_current_session_id()
    print(f"session_id before augment: {session_id}")
    print(f"keys: {keys}")
    summarized_embeddings = []
    augmented_embeddings = []
    if session_id:
        print("Does this run when not session_id")
        session_vector_store = get_session_vector_store(session_id, pre_delete_collection = False)
        persistent_vector_store = get_persistent_vector_store()
        num_keys = 0
        for key in keys: # each key here is analogous to each unique_id
            print(f"key_{num_keys}: {key}")
            num_keys += 1
            embeddings_for_file_with_key = get_embedding_from_manager(key) # embeddings_for_file_with_key = List[List[float]]
            # trimmed_print("embeddings_for_file_with_key", embeddings_for_file_with_key, 10000)
            # trimmed_print("fucking embeddings: ", embeddings, 10000)
            if embeddings_for_file_with_key:
                for embedding in embeddings_for_file_with_key:
                    # trimmed_print("fucking_embedding: ", embedding, 10000) // This is proven to be different for different keys
                    summarized_embedding = session_vector_store.similarity_search_by_vector(embedding=embedding, k=1)
                    # augmented_embedding = persistent_vector_store.similarity_search_with_relevance_scores( # return type == List[Tuple(Document, float)]
                    #     embedding=embedding,
                    #     k=1,
                    #     score_threshold=0.85
                    # )
                    # Also fetch additional context for each embedding from existing persistent vector store
                    # trimmed_print("fucking summarized embedding for current file: ", summarized_embedding, 10000)
                    summarized_embeddings.append(summarized_embedding)
                    # augmented_embeddings.append(augmented_embedding)
    print(f"summarized embeddings length: {len(str(summarized_embeddings))}")
    # trimmed_print("uploaded file to augment", summarized_embeddings, 10000)
    # summarized_embeddings.extend(augmented_embeddings)
    trimmed_print("fucking final context", summarized_embeddings, 10000)
    return summarized_embeddings

fetch_context = True # Whether or not to fetch context; defaults to true. It is global because the return type of get_question must be static in order for ordered_compression_retreiver to pipe from it
def get_question(inputs):
    """Method to get question"""
    # Get user question
    prompt: str = inputs['prompt']
    
    # Get whether or not to fetch context
    global fetch_context
    fetch_context = inputs['fetch_context']
    
    return prompt # Return type must be str to be able to pipe to retrieve_context_from_question

def get_question_and_return_as(inputs):
    """Method to get question"""
    # Get user question
    question: str = inputs['question']
    
    # Get whether or not to fetch context
    global fetch_context
    fetch_context = inputs['fetchContext']
    
    return question # Return type must be str to be able to pipe to retrieve_context_from_question
    
def retrieve_context_from_question(question):
    """Method to retrieve context from question"""
    global fetch_context # Get global fetch_context to see whether the user had set it to False
    if fetch_context:
        return ordered_compression_default_retriever # This guy only pipes from str
    else:
        return

def get_context_and_trace(dictionary, key):
    # Print the stack trace when this function is called
    # print(f"fucking docs context stacktrace: {traceback.print_stack()}")
    context = dictionary[key]
    # for document in context:
        # trimmed_print("fucking docs context", document, 1000)
    print(f"context length: {len(str(context))}")
    # trimmed_print("fucking context", context, 10000)
    return context

class Prompt(BaseModel):
    id: str
    content: str
    
def load_prompts() -> dict:
    if os.path.exists(PROMPTS_DATA_FILE):
        with open(PROMPTS_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
    
def save_prompts(prompts_dict: dict):
    with open(PROMPTS_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts_dict, f, indent=4)

def upload_prompt(prompt: Prompt):
    prompts = load_prompts()
    prompts[prompt.id] = prompt.dict()
    save_prompts(prompts)
    return {"message": "Prompt added successfully to json of prompts"}

def get_prompt_data(prompt_id):
    # Paths to the JSON files
    prompt_file = f"./database/prompt_{prompt_id}.json"
    locations_file = f"./database/locations_{prompt_id}.json"
    
    # Read and parse the job posting embeddings
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)
    
    # Read and parse the job URLs
    with open(locations_file, "r", encoding="utf-8") as f:
        locations_dict = json.load(f)
    
    # Retrieve the embeddings and URLs using job_id
    prompt_embedding = prompt_dict.get(prompt_id)
    location_list = locations_dict.get(prompt_id) # list of dicts where each dict is a location object

    # Return the tuple of job_posting_embedding and profile_urls
    return prompt_embedding, location_list

def retrieve_top_10_attractions(session_id: str, user_prompt_embeddings: List[List[float]], preprocessing_results):
    # Convert the list into a dictionary, preprocessing_results must never be empty
    jsonname_to_location_dict = {}
    for item in preprocessing_results:
        if item and item.get('result'):
            result = item['result']
            keys = list(result.keys())
            values = list(result.values())
            if keys and values:
                jsonname_to_location_dict[keys[0]] = values[0]
                
    print(f"location_json_to_location_object_dict: {jsonname_to_location_dict}")
    
    pgvectorstore = get_session_vector_store(session_id)
    session_vector_store = PGVectorWithEmbeddings(pgvectorstore)
    unique_attractions_embeddings: {str : List[List[float]]} = {}
    
    # To prevent repeats
    unique_attractions = []
    seen_paths = set()
    extracted = False
    
    for user_prompt_embedding in user_prompt_embeddings:
        if extracted:
            break
        
        # Perform similarity search
        results = session_vector_store.similarity_search_with_score_by_vector(user_prompt_embedding, k=3)
        print(f"retrieved results: {results}")
        if results:
            for doc, embedding, _ in results:
                try:
                    # print(f"doc: {doc}")
                    path = doc.metadata['source']
                    basename = os.path.basename(path) # profile pdf basename
                    # Fetch profile object from dict using basename
                    location = jsonname_to_location_dict.get(basename)
                    # Fetch id of profile
                    id = location['id']
                    
                    if path not in seen_paths:
                        # append profile to unique profiles
                        unique_attractions.append(location)
                        unique_attractions_embeddings.append(embedding)
                        seen_paths.add(path)
                    if len(unique_attractions) >= 11:
                        extracted = True
                        break
                    
                    if id not in unique_attractions_embeddings:
                        unique_attractions_embeddings[id] = []
                    unique_attractions_embeddings[id].append(embedding.tolist())
                except Exception as e:
                    print(f"Fuck {e}")
            
    attractions_embeddings_data_file = f"./database/attractions_embeddings_{session_id}.json"
    with open(attractions_embeddings_data_file, "w", encoding="utf-8") as f:
        json.dump(unique_attractions_embeddings, f)
        
    # Reset global session vector store and embeddings storage
    reset_session_vector_store(session_id=session_id)
    reset_embeddings_storage()
    
    if extracted:
        print("terminated in the middle")
        return unique_attractions[:-1]
    
    print(f"number of extracted unique attractions: {len(unique_attractions)}")
    return unique_attractions, list(unique_attractions_embeddings.values())

PROMPTS_DATA_FILE = "./database/prompts.json"

executor = ThreadPoolExecutor(max_workers=16)
# async def start_extracting(pipeline_data):
#     print("Start Extracting Locations")
#     prompt_id = pipeline_data['id']
#     type = pipeline_data['type']
#     prompt_embedding, raw_location_list = get_prompt_data(prompt_id)
    
#     initial_inputs = [
#         {
#             "instruction": "Extract relevant location properties given your context",
#             "raw_location_object": location.values()[0],
#             "id": location.keys()[0],
#             "type": type,
#             "session_id": prompt_id
#         }
#         for location in raw_location_list[1::2]
#     ]
    
#     async def create_task(input_data):
#         return await location_generation_chain.ainvoke(input_data)
    
#     def chunk_list(lst, n):
#         for i in range(0, len(lst), n):
#             yield lst[i:i + n]
    
#     num_chunks = 1
#     chunk_size = (len(initial_inputs) + num_chunks - 1) // num_chunks
#     input_chunks = list(chunk_list(initial_inputs, chunk_size))
    
#     async def event_generator():
#         preprocessing_results = []
#         retrieved_attractions = []
        
#         try:
#             for input_chunk in input_chunks:
#                 tasks = [create_task(input_data) for input_data in input_chunk]                
#                 loop = asyncio.get_event_loop()
#                 chunk_results = await loop.run_in_executor(executor, lambda: asyncio.run(asyncio.gather(*tasks)))
#                 preprocessing_results.extend(chunk_results)
                
#             retrieved_attractions, retrieved_attractions_embeddings = retrieve_top_20_attractions(session_id=prompt_id, user_prompt_embeddings=prompt_embedding, preprocessing_results=preprocessing_results)
#             unique_attractions_dict = {list(attraction.values())[0]: attraction for attraction in retrieved_attractions}
#             attraction_data_file = f"./database/attractions_{prompt_id}.json"
#             with open(attraction_data_file, "w", encoding="utf-8") as f:
#                 json.dump(unique_attractions_dict, f)
            
#             yield json.dumps({"event": "data", "id": prompt_id, "relevant_attractions": retrieved_attractions, "relevant_attractions_embeddings": retrieved_attractions_embeddings})
            
#         except Exception as e:
#             raise f"An error occured while processing prompt {prompt_id}: {e}"

#     return EventSourceResponse(event_generator(), media_type="text/event-stream") # media_type="text/event-stream"

async def start_extracting(pipeline_data):
    print(f"Start Extracting Locations: {pipeline_data}")
    prompt_id = pipeline_data['id']
    print(f"prompt id in start extracting: {prompt_id}")
    prompt_embedding, raw_location_list = get_prompt_data(prompt_id)
    # print(f"start extracting raw location list: {raw_location_list}")
    print(f"start_extracting prompt embedding: {prompt_embedding}")
    
    # input list
    initial_inputs = [
        {
            "instruction": "Extract relevant location properties given your context",
            "raw_location_object": list(location.values())[0],
            "id": list(location.keys())[0],
            "type": "generation",
            "session_id": prompt_id
        }
        for location in raw_location_list[1::2]
    ]
    print(f"initial inputs for start extracting: {initial_inputs}")
    
    # Function to create tasks for each input
    async def create_task(input_data):
        return await location_generation_chain.ainvoke(input_data)
    
    # Helper function to divide a list into chunks
    def chunk_list(lst, n):
        """Divide lst into n roughly equal parts"""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
            
    # Divide initial_inputs into 16 roughly equal parts
    num_chunks = 4
    chunk_size = (len(initial_inputs) + num_chunks - 1) // num_chunks
    print(f"chunk size: {chunk_size}")
    input_chunks = list(chunk_list(initial_inputs, chunk_size))
    
    preprocessing_results = []
    retrieved_attractions = []
    
    try:    
        for input_chunk in input_chunks:
            tasks = [create_task(input_data) for input_data in input_chunk]                
            chunk_results = await asyncio.gather(*tasks)
            preprocessing_results.extend(chunk_results)
            time.sleep(3)
    except Exception as e:
        raise f"An error occured while processing prompt {prompt_id}: {e}"

    # profile preprocessing results
    print(f"preprocessing results: {preprocessing_results}")
    
    retrieved_attractions, retrieved_attractions_embeddings = retrieve_top_10_attractions(session_id=prompt_id, user_prompt_embeddings=prompt_embedding, preprocessing_results=preprocessing_results)
    unique_attractions_dict = {list(attraction.values())[0]: attraction for attraction in retrieved_attractions}
    attraction_data_file = f"./database/attractions_{prompt_id}.json"
    with open(attraction_data_file, "w", encoding="utf-8") as f:
        json.dump(unique_attractions_dict, f)
    
    return {"embeddings_list": retrieved_attractions_embeddings, "session_vector_store": get_session_vector_store(prompt_id, False)}
    
def convert_embeddings_to_docs(embeddings_list: List[List[List[float]]], session_vector_store):
    # print(f"convert_embeddings_to_docs called! with session vector store: {session_vector_store}\n with embeddings: {embeddings_list}")
    automator = Automator()
    documents: List[List[Document]] = []
    
    for embeddings in embeddings_list:
        if embeddings is not None:
            documents.extend(automator.convert_embeddings_to_documents(embeddings, session_vector_store))
        
    return documents

async def embed_content(inputs): # Embeds either user prompt or saved txt objects of location JSON
    """Extension of run_google_searches to download PDFs and embed them."""
    print(f"embed_content inputs: {inputs}")
    embedding_method = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), show_progress_bar=True, chunk_size=10, skip_empty=True)
    
    # id = inputs['session_id'] # prompt_id or location_id, when its value is used, it is always location_id
    prompt = None
    type = inputs['type']
    id = None
    raw_location_object = None # Exclusive to generation type
    if type == "extraction":
        prompt = inputs['question']
        set_current_session_id(inputs['session_id'])
        file_embedder = FileEmbedder(inputs['session_id'])
    elif type == "generation":
        id = inputs['id']
        raw_location_object = inputs['raw_location_object']
        print(f"location session_id: {id}")
        file_embedder = FileEmbedder(id, pre_delete_collection=False)
    
    ### Initialize text splitter
    text_splitter = SemanticChunker(embeddings=embedding_method)
    
    try:
        embeddings_list = []
        
        ### Use unique id which was generated from React side
        unique_id = id
        upload_folder = Path("./uploaded_files")
        
        ### Vector Embedding
        # Embed website to session vector store
        file_extension = "txt"
        file_embedding = None
        file_path = None
        
        try:
            glob_pattern = f"**/*.{file_extension}"
            if type == "extraction":
                print(f"embed content Extraction type: {inputs['session_id']}")
                # Save User Prompt: str as txt to prompt_file_path that starts with f"./database/{unique_id}.txt"
                file_path = upload_folder / f"prompt_{inputs['session_id']}.txt"
                
                # Write the prompt to a text file
                try:
                    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                        await f.write(prompt)
                    print(f"Saved user prompt to: {file_path}")
                except Exception as e:
                    print(f"Failed to save user prompt: {e}")
                
                file_embedding = await file_embedder.process_and_embed_file(directory_path=upload_folder, file_path=file_path, unique_id=unique_id, url=None, embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter) # unique_id here is randomized prompt_id
            elif type == "generation":
                # Save Review: Dict as txt to review_file_path that starts with f"./database/attraction_review_{unique_id}.txt"
                file_path = upload_folder / f"location_object_{unique_id}.txt"
                
                # Write the review (dict) to a text file
                try:
                    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                        await f.write(json.dumps(raw_location_object))
                    print(f"Saved review to: {file_path}")
                except Exception as e:
                    print(f"Failed to save review: {e}")
                
                file_embedding = await file_embedder.process_and_embed_file(directory_path=upload_folder, file_path=file_path, unique_id=unique_id, url=None, embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter) # unique_id here is extracted location_id
        except Exception as e:
            raise f"Error processing content: {e}"
        finally:
            # Delete the file after processing it to avoid redundancy
            file_path.unlink(missing_ok=True)  # Use missing_ok=True to ignore the error if the file doesn't exist
        
        # Append to embeddings
        embeddings_list.append(file_embedding)
        
        # IFF type == extraction
        if type == "extraction":
            # Save the dictionary that uses unique_id as key and embeddings_list as value as json to system to mimic persistence
            prompt_data_file = f"./database/prompt_{inputs['session_id']}.json"
            prompt_dict = {inputs['session_id']: file_embedding}
            async with aiofiles.open(prompt_data_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(prompt_dict))
        
        # print(f"embeddings list: {embeddings_list}")
        
        # Return PDF embeddings_list: List[List[List[float]]] and session vector store for entire research session
        return {"embeddings_list": embeddings_list, "session_vector_store": get_session_vector_store(inputs['session_id'], False)}
    except HttpError as error:
        print(f"An error occurred: {error}")
    print("\n----------------------------------------\n")
        
class WrappedResponse:
    def __init__(self, llm_response, additional_data):
        self.llm_response = llm_response
        self.additional_data = additional_data

def generate_prompt(pipeline_data, type):
    print(f"Generate Prompt Called! {type}")
    # Select the appropriate template based on the type
    if type == "extraction":
        template = ARGUMENT_SCRAPING_TEMPLATE
    elif type == "generation":
        template = LOCATION_GENERATION_TEMPLATE
    else:
        raise ValueError("Invalid type. Must be either 'extraction' or 'generation'.")
    
    prompt_template = ChatPromptTemplate.from_template(template=template)
    prompt_output = prompt_template.format(**pipeline_data)
    if type == "generation":
        print(f"pipeline data for location: {pipeline_data}")
    else:
        id = pipeline_data['id']
        prompt = pipeline_data['prompt']
        prompt_object = Prompt(id=id, content=prompt)
        # Upload prompt and append to JSON of prompts
        upload_prompt(prompt_object)
    # Wrap the prompt output in the expected message format
    human_message = HumanMessage(content=prompt_output)
    print("Generate Prompt Successful!")
    return {
        "prompt_output": [human_message],  # LLM expects a list of messages
        "context": pipeline_data,  # Preserve the original context
        "type": type
    }
    
def call_llm_with_context(prompt_data):
    # Select the appropriate LLM based on the type
    model_type = "llm_tourpedia_api_arguments_extractor"
    if prompt_data['type'] == "extraction":
        llm_model = llm_tourpedia_api_arguments_extractor
    elif prompt_data['type'] == "generation":
        llm_model = llm_tourpedia_location_object_generator
        model_type = "llm_tourpedia_location_object_generator"
    else:
        raise ValueError("Invalid type. Must be either 'extraction' or 'generation'.")
    print(f"Successfully chose llm_model in call_llm_with_context: {model_type}")
    # print(f"chosen llm_model: {llm_model.__str__()}")
    llm_input = prompt_data["prompt_output"]
    # print(f"llm input: {llm_input}")
    # print(f"typeof llm input: {type(llm_input)}")
    context = prompt_data["context"]
    llm_response = llm_model(llm_input)
    print(f"Successfully received llm_response in call_llm_with_context! {llm_response}")
    wrapped_response = WrappedResponse(llm_response=llm_response, additional_data=context)
    return wrapped_response

extractor_global = Extractor()
def call_populated_function(wrapped_response):
    llm_response = wrapped_response.llm_response
    additional_data = wrapped_response.additional_data
    
    try:
        additional_kwargs = llm_response.additional_kwargs if hasattr(llm_response, 'additional_kwargs') else {}
        tool_calls = additional_kwargs.get('tool_calls', [])
        if tool_calls:
            tool_call = tool_calls[0]
            if tool_call['function']['name'] == "extract_relevant_api_arguments":
                arguments = json.loads(tool_call['function']['arguments'])
                location = arguments["location"]
                category = arguments["category"]
                tourpedia_api_arguments = extractor_global.extract_relevant_api_arguments(
                    location=location, category=category
                )
                additional_data['tourpedia_api_arguments'] = tourpedia_api_arguments
                
                print(f"additional data id in call populated function: {additional_data['id']}")
                
                return {additional_data['id'] : tourpedia_api_arguments}
            elif tool_call['function']['name'] == "extract_relevant_location_properties":
                extractor = Extractor()
                arguments = json.loads(tool_call['function']['arguments'])
                
                name = arguments["name"]
                address = arguments["address"]
                url = arguments["url"]
                pros = arguments["pros"]
                cons = arguments["cons"]
                rating = arguments["rating"]
                
                location_with_properties = extractor.extract_relevant_location_properties(
                    id=additional_data['id'], name=name, address=address, url=url, pros=pros,  cons=cons, rating=rating
                )
                additional_data['location_with_properties'] = location_with_properties
                
                return location_with_properties
        return wrapped_response
    except Exception as e:
        print(f"An error occurred in call_populated_function: {e}")
        return None
    
def getIDs(args):
    idList = []
    cityList = ["Berlin", "Amsterdam", "Barcelona", "Dubai", "London", "Paris", "Rome", "Tuscany"]
    catList = ["attraction", "restaurant", "poi"]

    # user input capitalized and trailing whitespace removed
    city = args[0].capitalize().rstrip()
    # check if the city is valid input
    if city in cityList:
        pass
    else:
        print("Invalid city")
        exit()

    # user input in lowercase and trailing whitespace removed
    cat = args[1].lower().rstrip()
    # check if the city is valid input
    if cat in catList:
        pass
    else:
        print("Invalid category")
        exit()


    url = "http://tour-pedia.org/api/getPlaces?category="+cat+"&location="+city
    myResponse = None
    try:
        # Send a GET request with a specified timeout (e.g., 10 seconds)
        myResponse = requests.get(url, timeout=10)
        # Check if the response was successful (status code 200)
        myResponse.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")

    # For successful API call, response code will be 200 (OK)
    if(myResponse.ok):
        # Loading the response data into a dict variable
        # json.loads takes in only binary or string variables so using content to fetch binary content
        # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
        jData = json.loads(myResponse.content)
        # Iterate directly through the list of dictionaries
        for place in jData:
            # Iterate through the key-value pairs in each dictionary
            for key, value in place.items():
                if key == "id":
                    idList.append(value)
    else:
        # If responsecode is not ok (200), print the resulting http error code with description
        myResponse.raise_for_status()
    return idList

executor = ThreadPoolExecutor(max_workers=8)  # Adjust max_workers as needed
# def fetch_reviews_for_id(id):
#     # Dictionary to store results for the specific ID
#     locations_raw_object_dict = {}
#     reviews = []
   
#     # Fetch reviews
#     url = f"http://tour-pedia.org/api/getReviewsByPlaceId?language=en&placeId={str(id)}"
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         jData = json.loads(response.content)
#         reviews = [review['text'] for review in jData if 'text' in review]  # Extract reviews


#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching reviews for {id}: {e}")
#         return {id: {"error": str(e)}}
   
#     # if the reviews list has elements, if not, that id is not added to the dict
#     if len(reviews) != 0:
#         # store reviews
#         locations_raw_object_dict[id] = [reviews]
       
#         # Fetch place details
#         placeName = ""
#         placeAddy = ""
#         details_url = f"http://tour-pedia.org/api/getPlaceDetails?id={str(id)}"
#         try:
#             response = requests.get(details_url, timeout=10)
#             response.raise_for_status()
#             jData = json.loads(response.content)
#             placeName = jData.get("name", "")
#             placeAddy = jData.get("address", "")


#         except requests.exceptions.RequestException as e:
#             print(f"Error fetching details for {id}: {e}")
#             return {id: {"error": str(e)}}
       
#         # Store place details
#         locations_raw_object_dict[id].append(placeName)
#         locations_raw_object_dict[id].append(placeAddy)
#         locations_raw_object_dict[id].append(details_url)
   
#     return locations_raw_object_dict

def fetch_reviews_for_id(id):
    # Dictionary to store results for the specific ID
    locations_raw_object_dict = {}
    reviews = []
    
    # Fetch reviews
    url = f"http://tour-pedia.org/api/getReviewsByPlaceId?language=en&placeId={str(id)}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        jData = json.loads(response.content)
        reviews = [review['text'] for review in jData if 'text' in review]  # Extract reviews

    except requests.exceptions.RequestException as e:
        print(f"Error fetching reviews for {id}: {e}")
        return {id: {"error": str(e)}}
    
    # Store reviews
    locations_raw_object_dict[id] = [reviews]
    
    # Fetch place details
    placeName = ""
    placeAddy = ""
    details_url = f"http://tour-pedia.org/api/getPlaceDetails?id={str(id)}"
    try:
        response = requests.get(details_url, timeout=10)
        response.raise_for_status()
        jData = json.loads(response.content)
        placeName = jData.get("name", "")
        placeAddy = jData.get("address", "")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching details for {id}: {e}")
        return {id: {"error": str(e)}}
    
    # Store place details
    locations_raw_object_dict[id].append(placeName)
    locations_raw_object_dict[id].append(placeAddy)
    locations_raw_object_dict[id].append(details_url)
    
    return locations_raw_object_dict

def getReviews(idList):
    locations_raw_object_dict = {}
    
    # Create tasks for fetching data in parallel
    futures = [executor.submit(fetch_reviews_for_id, id) for id in idList[:50]]
    
    # Process results as they are completed
    for future in as_completed(futures):
        result = future.result()
        if result:
            print(f"response for fetch raw location objection for id: {result}")
            locations_raw_object_dict.update(result)
    
    return locations_raw_object_dict

# def getReviews(idList):
#     print("Get Reviews Called")
#     locations_raw_object_dict = {}
#     reviews = []

#     for id in idList:
#         url = f"http://tour-pedia.org/api/getReviewsByPlaceId?language=en&placeId={str(id)}"
#         myResponse = None
#         try:
#             # Send a GET request with a specified timeout (e.g., 10 seconds)
#             myResponse = requests.get(url, timeout=10)
#             # Check if the response was successful (status code 200)
#             myResponse.raise_for_status()
#         except requests.exceptions.HTTPError as http_err:
#             print(f"HTTP error occurred: {http_err}")
#         except requests.exceptions.ConnectionError as conn_err:
#             print(f"Connection error occurred: {conn_err}")
#         except requests.exceptions.Timeout as timeout_err:
#             print(f"Timeout error occurred: {timeout_err}")
#         except requests.exceptions.RequestException as req_err:
#             print(f"An error occurred: {req_err}")
#         # For successful API call, response code will be 200 (OK)
#         if(myResponse.ok):
#             # Loading the response data into a dict variable
#             # json.loads takes in only binary or string variables so using content to fetch binary content
#             # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
#             jData = json.loads(myResponse.content)
#             # initialize the list of review strings for the id to be null
#             locations_raw_object_dict[id] = []
#             reviews = []
#             # Iterate directly through the list of dictionaries
#             for review in jData:
#                 # Iterate through the key-value pairs in each dictionary
#                 for key, value in review.items():
#                     if key == "text":
#                         # add the text to the dictionary attacted to the placeID
#                         reviews.append(value)
#             # add all the String reviews as a element in the value attached to the placeID key
#             locations_raw_object_dict[id].append(reviews)
#             # place attributes to add to the dict
#             placeName = ""
#             placeAddy = ""
#             # get the rest of the data associated with the location and add it to the list
#             url = f"http://tour-pedia.org/api/getPlaceDetails?id={str(id)}"
#             myResponse = None
#             try:
#                 # Send a GET request with a specified timeout (e.g., 10 seconds)
#                 myResponse = requests.get(url, timeout=10)
#                 # Check if the response was successful (status code 200)
#                 myResponse.raise_for_status()
#             except requests.exceptions.HTTPError as http_err:
#                 print(f"HTTP error occurred: {http_err}")
#             except requests.exceptions.ConnectionError as conn_err:
#                 print(f"Connection error occurred: {conn_err}")
#             except requests.exceptions.Timeout as timeout_err:
#                 print(f"Timeout error occurred: {timeout_err}")
#             except requests.exceptions.RequestException as req_err:
#                 print(f"An error occurred: {req_err}")
#             if(myResponse.ok):
#                 jData = json.loads(myResponse.content)
#                 # Iterate through the key-value pairs in each dictionary
#                 for key, value in jData.items():
#                     if key == "name":
#                         placeName = value
#                     elif key == "address":
#                         placeAddy = value
#             else:
#                 # If response code is not ok (200), print the resulting http error code with description
#                 myResponse.raise_for_status()
#             locations_raw_object_dict[id].append(placeName)
#             locations_raw_object_dict[id].append(placeAddy)
#             locations_raw_object_dict[id].append(url)
            
#             print(f"response: {myResponse}")

#         else:
#             # If response code is not ok (200), print the resulting http error code with description
#             myResponse.raise_for_status()
#     # { placeID: [[review1,review2], placeName, placeAddress, placeURL], placeID2:[...]...}
#     return locations_raw_object_dict

async def return_raw_location_list(input):
    print("Return raw location")
    
    # Extract id and arguments
    id = list(input.keys())[0]
    arguments = list(input.values())[0]
    print(f"id: {id}")
    print(f"arguments: {arguments}")
    
    # Get list of IDs based on arguments
    id_list = getIDs(arguments)
    
    # Async wrapper for getReviews using ThreadPoolExecutor
    loop = asyncio.get_event_loop()

    # Define async function to handle getReviews call
    async def fetch_reviews_async(id_list):
        print("Start fetching reviews")
        location_raw_objects_dict = await loop.run_in_executor(executor, lambda: getReviews(id_list))
        print("Finished fetching reviews")
        return location_raw_objects_dict

    # Call the async getReviews and wait for the results
    location_raw_object_dict = await fetch_reviews_async(id_list)
    print(f"Get Reviews Success: {location_raw_object_dict}")

    # Convert dict to list where each item is a dict of a single key-value pair
    location_raw_object_list = [{location_id: raw_object} for location_id, raw_object in location_raw_object_dict.items()]

    # Save the results to a JSON file
    locations_for_prompt_file = f"./database/locations_{id}.json"
    raw_locations_dict = {id: location_raw_object_list}

    with open(locations_for_prompt_file, "w", encoding="utf-8") as f:
        json.dump(raw_locations_dict, f)
        
    print(f"location raw object list: {location_raw_object_list}")

    return location_raw_object_list

# Function to run async task from a non-async context
def run_return_raw_location_list(input):
    return asyncio.run(return_raw_location_list(input))
    
# def return_raw_location_list(input):
#     print("Return raw location ")
#     id = list(input.keys())[0]
#     arguments = list(input.values())[0]
#     print(f"id: {id}")
#     print(f"arguments: {arguments}")
#     # Douglass's Function
#     id_list = getIDs(arguments)
#     location_raw_object_list = getReviews(id_list) # dict of each key as location_id and value as raw location object
#     print("Get Reviews Success")
#     # Save raw_locations_dict as json for future fetching with prompt_id
#     locations_for_prompt_file = f"./database/locations_{id}.json"
#     raw_locations_dict = {id: location_raw_object_list}
#     with open(locations_for_prompt_file, "w", encoding="utf-8") as f:
#         json.dump(raw_locations_dict, f)
#     return location_raw_object_list

# Correctly setting up partial functions
custom_question_getter = partial(get_question)
question_context_retriever_runnable = RunnableLambda(lambda question_runnable_output: retrieve_context_from_question(question_runnable_output))
custom_context_getter = partial(get_context_and_trace, key="context") # change to context
custom_augment_getter = partial(get_augment_and_trace)

return_content_embeddings = partial(embed_content)
return_converted_docs = RunnableLambda(lambda data: convert_embeddings_to_docs(**data))

generate_prompt_runnable = RunnableLambda(lambda pipeline_data: generate_prompt(pipeline_data, pipeline_data['type']))
call_llm_with_context_runnable = RunnableLambda(lambda prompt_data: call_llm_with_context(prompt_data))
call_extract_relevant_data = RunnableLambda(lambda wrapped_response: call_populated_function(wrapped_response))
return_raw_location_list_runnable = RunnableLambda(lambda api_arguments: run_return_raw_location_list(api_arguments))
# start_extraction_runnable = RunnableLambda(lambda pipeline_data: start_extracting(pipeline_data))
start_extraction_runnable = RunnableLambda(lambda pipeline_data: asyncio.run(start_extracting(pipeline_data)))

# Extractor getPlaces api endpoint argument extraction chain; This is the endpoint that will receive user prompt
extract_arguments_chain = (
    RunnableParallel(
        # custom_question_getter should get the user prompt, then vector embed them into a List[List[float]]
        context=(return_content_embeddings | return_converted_docs), # context is a List[List[Document]] that is made by first vector embedding the user prompt and then converting it to Document objects
        prompt=itemgetter("question"),
        id=itemgetter("session_id"),
        type=itemgetter("type"),
        fetch_context = itemgetter("fetchContext"),
        file_embedding_keys=itemgetter("file_embedding_keys")
    ) |
    RunnableParallel(
        id=itemgetter("id"),
        file_embedding_keys=itemgetter("file_embedding_keys"),
        prompt=custom_question_getter,
        results=(generate_prompt_runnable | call_llm_with_context_runnable | call_extract_relevant_data | return_raw_location_list_runnable) # type(return_raw_location_list_runnable) == List[Dict[locationId, rawLocationDict]] type(call_extract_relevant_data) == List[str] where each str is an argument for the API call.
        # augment=lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
    ) |
    RunnableParallel(
        # Pipe result directly to start_extracting which needs its final output to be a List[List[Document]] for extracted locations
        file_embedding_keys=itemgetter("file_embedding_keys"),
        context=(start_extraction_runnable | return_converted_docs), # Return type of start_extracting == List[Dict[placeId, place_object]]; Turn these review entities to a combined List[List[float]] and pass it to return_converted_docs to result in List[List[Dict]]
        prompt=itemgetter("prompt")
    ) |
    RunnableParallel (
        answer = (ChatPromptTemplate.from_template(template=LOCATION_GUIDE_TEMPLATE) | llm_conversation),
        docs = custom_context_getter,
        augment = lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
    )
).with_types(input_type=dict)

# Extractor location generation chain; Called in parallel; arguments to each chain 
location_generation_chain = (
    RunnableParallel(
        context=(return_content_embeddings | return_converted_docs), # turn review objects into a List[List[Document]]
        instruction=itemgetter("instruction"),
        id=itemgetter("id"),
        type=itemgetter("type"),
        raw_location_object=itemgetter("raw_location_object") # review == dict
    ) |
    RunnableParallel(
        result=(generate_prompt_runnable | call_llm_with_context_runnable | call_extract_relevant_data) # result must be a Dict with the location id as the key and generated location object as the value.
    )
).with_types(input_type=dict)

# Default Chat Conversation Chain
final_chain = (
    RunnableParallel (
        context=(custom_question_getter | question_context_retriever_runnable),
        augment = custom_augment_getter, # type of augment is List[List[Document]]
        question=itemgetter("question"), # type of itemgetter("question") == string
        file_embedding_keys=itemgetter("file_embedding_keys") # Optional
    ) |
    RunnableParallel (
        answer = (ChatPromptTemplate.from_template(template=CONVERSATION_TEMPLATE) | llm_conversation),
        docs = custom_context_getter,
        augment = lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
        augments = itemgetter("augment")
    )
).with_types(input_type=RagInput)

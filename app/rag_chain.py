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

from app.embeddings_manager import get_current_session_id, get_embedding_from_manager, get_embeddings_storage, get_persistent_vector_store, get_session_vector_store, EMBEDDINGS_STORAGE, reset_embeddings_storage, reset_session_vector_store, set_current_session_id, get_driver
from app.config import PG_COLLECTION_NAME, EMBEDDING_MODEL

from importer.load_and_process import chatgpt_base
from importer.load_and_process import FileEmbedder

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

SOLVE_TEMPLATE = """
Answer given the following:

Current problem: 
{problem}

Total problems: 
{persistent_cell_context}

Solved problems: 
{previous_cell_context}

Write all math symbols in latex such that they can be executed in a Jupyter Notebook
"""

PREPROCESSING_TEMPLATE = """
Extract a full list of problems that need to be solved from the augment:

Uploaded file augment: 
{augment}

Retrieved context: 
{context}

Question from: 
{question}

Note:
- If a single Jupyter notebook contains the problems that need to be solved, extract all relevant problems from the notebook.
- If dealing with a PDF with a comprehensive list of problems AND a PDF that lists the problems that need to be solved is NOT uploaded, extract all relevant problems from the PDF with a comprehensive list of problems.
- If a PDF that points to another document for problems is uploaded, identify and extract ONLY the relevant problems from the referred document.

Each problem requires a label for both the problem number it is part of and sub problem it is part of.
"""

RESEARCH_TEMPLATE = """
Explain academic papers given the following:

Retrieved context:
{context}

Problem from: {question}
"""

WEB_SCRAPING_TEMPLATE = """
Extract relevant Linkedin API search tags from the given job posting page given the following:

Detailed Prompt: 
{instruction}

Vector Embedded Job Posting Page: 
{context}

File Embedding Keys:
{file_embedding_keys}

ID:
{id}

Return the extracted search tags as arguments to a function call
"""

PROFILE_SCRAPING_TEMPLATE = """
Extract relevant profile keywords from the given profile page given the following:

Detailed Prompt: 
{instruction}

Vector Embedded Profile Page: 
{context}

Return the extracted keywords as arguments to a function call
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

def get_pdf_url_from_page(abs_url: str) -> str:
    """Extracts the PDF download URL from a given page URL in a generic way.

    Args:
        abs_url (str): URL of the page to extract the PDF URL from.

    Returns:
        str: PDF download URL if found, otherwise None.
    """
    try:
        # Send a GET request with a specified timeout (e.g., 10 seconds)
        response = requests.get(abs_url, timeout=10)
        # Check if the response was successful (status code 200)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
        return None
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
        return None

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all anchor tags
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Check if 'pdf' is in the href attribute
        if 'pdf' in href:
            # Build the full URL if it's relative
            pdf_url = urljoin(abs_url, href)
            return pdf_url

    return None

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

def redirect_test(dictionary, key):
    # Print the stack trace when this function is called
    # print(f"fucking docs context stacktrace: {traceback.print_stack()}")
    question = dictionary[key]
    # for document in context:
        # trimmed_print("fucking docs context", document, 1000)
    print(f"redirected to solver!: {len(str(question))}")
    return question

# def process_response(dictionary, key):
#     """Method that processes initial LLM response into a format that can be received by JupyterSolver.process_files_and_generate_response"""
#     response = dictionary[key]
#     print(f"process_response called: {response}")
    
#     # Assuming 'response' is an instance of AIMessageChunk and 'additional_kwargs' is an attribute of it
    # if hasattr(response, 'additional_kwargs'):
    #     tool_calls = response.additional_kwargs.get('tool_calls', [])
    # else:
    #     tool_calls = []
    # print(f"tool_calls: {tool_calls}")
    
#     extracted_data = {}
#     for tool_call in tool_calls:
#         if 'function' in tool_call and tool_call['function'].get('name') == "process_files_and_generate_response":
#             arguments_json = tool_call['function'].get('arguments', '{}')
#             arguments_dict = json.loads(arguments_json)
#             # Directly extracting 'problem_file_types' and 'problem_files'
#             problem_file_types = arguments_dict.get("dictionary", {}).get("problem_file_types", [])
#             problem_files = arguments_dict.get("dictionary", {}).get("problem_files", [])
#             extracted_data = {
#                 "problem_file_types": problem_file_types,
#                 "problem_files": problem_files
#             }
#             print(f"Extracted Data: {extracted_data}")
#             break
#     else:
#         extracted_data = "Error: function process_files_and_generate_response does not exist"
    
#     return extracted_data # Should be Dict that has 2 keys: problem_file_types, problem_files; for problem_file_types the value is a List of file types in .extension form, for problem_files, the value is a List of actual Document objects that contain the problems that need to be solved.

# def process_response(dictionary, key):
#     """Extract questions, problem file path, and pointer file path from the LLM response."""
#     response = dictionary[key]
#     print(f"process_response called: {response}")

#     # Initialize default values
#     problems = []
#     problem_file_path = ""
#     pointer_file_path = ""

#     if 'tool_calls' in response:
#         for tool_call in response['tool_calls']:
#             if tool_call['function']['name'] == "solve_problems":
#                 arguments_dict = tool_call['function']['arguments']
#                 problems = arguments_dict.get("problems", [])
#                 problem_file_path = arguments_dict.get("problem_file_path", "")
#                 pointer_file_path = arguments_dict.get("pointer_file_path", "")
#                 print(f"Extracted: Problems - {problems}, Problem File Path - {problem_file_path}, Pointer File Path - {pointer_file_path}")
#                 break
#         else:
#             print("Error: function solve_problems does not exist or no questions extracted.")
#     else:
#         print("No tool calls found in response.")

#     return problems, problem_file_path, pointer_file_path

def preprocess_json_string(arguments_json):
    """Preprocess JSON string to escape problematic characters safely."""
    # Escaping backslashes first to avoid double escaping
    arguments_json = arguments_json.replace('\\', '\\\\')
    
    # Escaping double quotes within the string, ensuring not to escape legitimate JSON structure
    # This simplistic approach might not be foolproof for all edge cases
    arguments_json = re.sub(r'(?<!\\)"', '\\"', arguments_json)
    
    return arguments_json

def process_response(dictionary, key):
    """Extract questions, problem file path, and pointer file path from the LLM response."""
    response = dictionary[key]
    problems = []
    problem_file_path = ""
    pointer_file_path = ""
    print(f"response: {response}")
        
    if hasattr(response, 'additional_kwargs'):
        tool_calls = response.additional_kwargs.get('tool_calls', [])
    else:
        tool_calls = []
    print(f"tool_calls: {tool_calls}")
    
    for tool_call in tool_calls:
        if tool_call['function']['name'] == "solve_problems":
            arguments_json = tool_call['function'].get('arguments', '{}')
            print(f"arguments_json: {arguments_json}")
            # Preprocessing the JSON string to escape problematic characters
            preprocessed_json = preprocess_json_string(arguments_json)
            # try:
            #     arguments_dict = json.loads(arguments_json)
            # except json.JSONDecodeError as e:
            #     print(f"Error decoding JSON: {e}")
            #     continue  # or handle error as appropriate
            
            problems = arguments_json.get("problems", [])
            problem_file_path = arguments_json.get("problem_file_path", "")
            pointer_file_path = arguments_json.get("pointer_file_path", "")
            print(f"Extracted: Problems - {problems}, Problem File Path - {problem_file_path}, Pointer File Path - {pointer_file_path}")
            break
        else:
            print("Error: function solve_problems does not exist or no questions extracted.")

    return problems, problem_file_path, pointer_file_path

async def fetch_context_document_with_score(inputs, retriever=ordered_compression_score_retriever):
    search_query = inputs['question']
    results = await retriever.base_retriever.vectorstore.asimilarity_search_with_relevance_scores( # return type == List[Tuple(Document, float)]
        search_query, 
        9, 
        score_threshold=0.95
    )
    run_searches = False
    if not results:
        run_searches = True
        return results, run_searches
    
    for result in results:
        print(f"similarity: {result[1]}")
    
    trimmed_print("trimmed similarity search with score: ", results, 1000)
    return results, run_searches

# Site restriction guiding prompt to search term
sites = ["site: https://arxiv.org"]
google_guidance_prompts = [f"{site} " for site in sites]

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build(serviceName="customsearch", version="v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    # print(res['items'])
    return res['items']
        
async def run_google_searches_and_embed_pdfs(inputs, api_key=google_api_key, cse_id=google_cse_id, guidance_prompts=google_guidance_prompts, embedding_method=embedding_method, fetch_num=3):
    """Extension of run_google_searches to download PDFs and embed them."""
    search_query = inputs['question']
    ### Initialize session_id and FileEmbedder instance before embedding
    session_id = random_base36_string()
    set_current_session_id(session_id) # set session id
    file_embedder = FileEmbedder(session_id)
    text_splitter = SemanticChunker(embeddings=embedding_method)
    
    for guidance_prompt in guidance_prompts:
        full_search_term = f"{guidance_prompt} {search_query}"
        try:
            results = google_search(full_search_term, api_key, cse_id, num=fetch_num)
            pdf_urls = []
            for result in results:
                if 'arxiv.org/abs/' in result['link']:
                    pdf_url = get_pdf_url_from_page(result['link'])
                    if pdf_url:
                        pdf_urls.append(pdf_url)
            print(f"pdf urls: {pdf_urls}")
            embeddings_list: List[List[List[float]]] = []
            
            ### For each of the 8 pdf urls, download the pdfs, open it, and embed to session vector store
            for url in pdf_urls:
                response = None
                try:
                    # Send a GET request with a specified timeout (e.g., 10 seconds)
                    response = requests.get(url, timeout=10)
                    # Check if the response was successful (status code 200)
                    response.raise_for_status()
                except requests.exceptions.HTTPError as http_err:
                    print(f"HTTP error occurred: {http_err}")
                except requests.exceptions.ConnectionError as conn_err:
                    print(f"Connection error occurred: {conn_err}")
                except requests.exceptions.Timeout as timeout_err:
                    print(f"Timeout error occurred: {timeout_err}")
                except requests.exceptions.RequestException as req_err:
                    print(f"An error occurred: {req_err}")
                
                ### Name Extraction
                # Unique ID for file
                unique_id = str(uuid.uuid4())
                # Extract file path with appropriate file name
                filename = get_filename_from_response(response, url)
                upload_folder = Path("./uploaded_files")
                save_path = upload_folder / f"{filename}"
                
                ### Writing content
                # Write response.content to pdf file
                with open(str(save_path), 'wb') as file:
                    file.write(response.content)
                
                ### Vector Embedding
                # Embed PDF file to session vector store
                file_extension = "pdf"
                file_embedding = None
                try:
                    glob_pattern = f"**/*.{file_extension}"
                    file_embedding = await file_embedder.process_and_embed_file(unique_id=unique_id, directory_path=upload_folder, file_path=save_path, embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter)
                except Exception as e:
                    print(f"Error processing file {filename}.pdf: {e}")
                finally:
                    # Delete the file after processing it to avoid redundancy
                    save_path.unlink(missing_ok=True)  # Use missing_ok=True to ignore the error if the file doesn't exist
                
                # Append to embeddings
                embeddings_list.append(file_embedding)
            
            # Return PDF embeddings_list: List[List[List[float]]] and session vector store for entire research session
            return {"embeddings_list": embeddings_list, "session_vector_store": get_session_vector_store(session_id)}
        except HttpError as error:
            print(f"An error occurred: {error}")
        print("\n----------------------------------------\n")
        
# def convert_embeddings_to_docs(embeddings_list: List[List[List[float]]], session_vector_store):
#     print(f"convert_pdf_embeddings_to_docs called! with session vector store: {session_vector_store}")
#     jupyter_solver = JupyterSolver()
#     documents: List[List[Document]] = []
    
#     for embeddings in embeddings_list:
#         if embeddings is not None:
#             documents.extend(jupyter_solver.convert_embeddings_to_documents(embeddings, session_vector_store))
        
#     return documents
            
# class CustomVectorStoreRetriever(VectorStoreRetriever):
#     """
#     A custom retriever that extends the basic VectorStoreRetriever to include
#     similarity scores and potentially the full content of documents in its results.
#     """
    
#     def __init__(self, store):
#         # super().__init__()
#         self = store.as_retriever(
#             search_type="similarity", search_kwargs={"k": 9, "include_metadata": True}
#         )

#     def _get_relevant_documents(self, query: str, *, run_manager) -> List[Tuple[Document, float]]:
#         """
#         Override to fetch documents along with their similarity scores.
#         """
#         if self.search_type == "similarity":
#             # Fetch documents and their similarity scores
#             docs_and_scores = self.max_marginal_relevance_search_with_score(query, **self.search_kwargs)
#             return docs_and_scores
#         else:
#             raise ValueError(f"Unsupported search type: {self.search_type}")

#     def get_documents_with_scores(self, query: str) -> List[Tuple[Document, float]]:
#         """
#         A public method to retrieve documents along with their similarity scores.
#         """
#         return self._get_relevant_documents(query, run_manager=None)

### ESCALATOR SHIT
class LinkedinInput(TypedDict, total=False):
    # In order for an input value from the frontend to be processed by the backend, it needs to be defined within an acceptable input class
        # Even optional inputs need to be defined!!
    instruction: str
    id: str
    url: str # url for the job applied or job user will apply
    file_embedding_keys: Optional[List[str]] # Optional list of strings
    
class ProfilePreprocessingInput(TypedDict, total=False):
    instruction: str
    id: str
    url: str
    
class ProfileRetrievalInput(TypedDict, total=False):
    instruction: str
    id: str
    url: str
    
# Function to copy the driver session
# def copy_driver_session(original_driver: webdriver.chrome.webdriver.WebDriver) -> webdriver.chrome.webdriver.WebDriver:
#     options = uc.ChromeOptions()
#     options.add_argument('--headless')
#     new_driver = uc.Chrome(options = options, use_subprocess = True)

#     # Retrieve cookies from the original driver
#     cookies = original_driver.get_cookies()
    
#     # Navigate to LinkedIn to set up the session
#     new_driver.get('https://www.linkedin.com')

#     # Add cookies to the new driver
#     for cookie in cookies:
#         new_driver.add_cookie(cookie)

#     # Navigate to LinkedIn feed page to ensure the session is maintained
#     new_driver.get('https://www.linkedin.com/feed/')
#     return new_driver

# webdriver.chrome.webdriver.WebDriver
def copy_driver_session(original_driver: uc.Chrome) -> uc.Chrome:
    options = uc.ChromeOptions()
    # options.add_argument('--headless') # IDK why but keeping the browser headless fucks up pdf downloading
    # Ensure that the new driver uses the virtual display
    # options.add_argument('--no-sandbox')  # Disables the sandbox for compatibility
    # options.add_argument('--disable-dev-shm-usage')  # Prevents using /dev/shm, useful in Docker
    # options.add_argument('--disable-gpu')  # Disables GPU hardware acceleration
    # options.add_argument('--disable-software-rasterizer')  # Disables software rasterizer
    # options.add_argument('--disable-extensions')  # Disables all Chrome extensions
    # options.add_argument('--disable-blink-features=AutomationControlled')
    # options.add_argument('--incognito')
    # options.add_argument('--no-default-browser-check')
    # options.add_argument('--no-first-run')
    
    ###
    # options.add_argument('--no-sandbox')
    # options.add_argument('--disable-infobars')
    # options.add_argument('--disable-dev-shm-usage')
    # options.add_argument('--disable-browser-side-navigation')
    # options.add_argument("--remote-debugging-port=9222")
    options.add_argument('--disable-gpu')
    # options.add_argument("--log-level=3")
    # print(f"display after vdisplay initialization: {os.getenv('DISPLAY')}")
    # options.add_argument(f'--display={os.getenv("DISPLAY")}')
    ###
    
    ###
    # options.add_argument('--headless')
    # options.add_argument("start-maximized")
    # options.add_argument('enable-automation')
    ###
    
    new_driver = uc.Chrome(options=options)

    # Retrieve cookies from the original driver
    cookies = original_driver.get_cookies()

    # Navigate to LinkedIn to set up the session
    new_driver.get('https://www.linkedin.com')

    # Add cookies to the new driver
    for cookie in cookies:
        print(f"fetched cookie from existing driver: {cookie}")
        new_driver.add_cookie(cookie)

    # Copy local storage
    # local_storage = original_driver.execute_script("return window.localStorage;")
    # for key, value in local_storage.items():
    #     new_driver.execute_script(f"window.localStorage.setItem('{key}', '{value}');")

    # Navigate to LinkedIn feed page to ensure the session is maintained
    new_driver.get('https://www.linkedin.com/feed/')
    
    new_driver.minimize_window()
    return new_driver

# async def embed_website(inputs):
#     """Extension of run_google_searches to download PDFs and embed them."""
#     print(f"embed_website inputs: {inputs}")
#     embedding_method = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), show_progress_bar=True, chunk_size=100, skip_empty=True)
    
#     id = inputs['id'] # job_id or profile_id
#     url = inputs['url']
#     type = inputs['type']
#     if type == "job":
#         session_id = random_base36_string()
#         set_current_session_id(session_id)
#     elif type == "profile":
#         session_id = inputs['session_id']
#         print(f"profile session_id: {session_id}")
    
#     ### Initialize session_id and FileEmbedder instance before embedding
#     file_embedder = FileEmbedder(session_id)
#     # How does this work without pre_delete_collection = False ?
#     # It works because this is for the initial creation of the session vector store and when adding documents we set pre_delete_collection to true
#     text_splitter = SemanticChunker(embeddings=embedding_method)
    
#     try:
#         embeddings_list: List[List[List[float]]] = []
        
#         ### Use unique id which was generated from React side which will be used to fetch embeddings for url
#         unique_id = id
#         upload_folder = Path("./uploaded_files")
        
#         ### Vector Embedding
#         # Embed website to session vector store
#         file_extension = "html"
#         file_embedding = None
#         try:
#             glob_pattern = f"**/*.{file_extension}"
#             if type == "job":
#                 file_embedding = await file_embedder.process_and_embed_file(directory_path=upload_folder, file_path=None, unique_id=unique_id, url=url, embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter)
#             elif type == "profile":
#                 new_driver = copy_driver_session(escalator_global.scraper.driver)
#                 file_embedding = await file_embedder.process_and_embed_file(driver=new_driver, directory_path=upload_folder, file_path=None, unique_id=unique_id, url=url, embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter)
#         except Exception as e:
#             raise f"Error processing url {url}: {e}"
        
#         # Append to embeddings
#         embeddings_list.append(file_embedding)
        
#         # IFF type == job
#         if type == "job":
#             # Save the dictionary that uses unique_id as key and embeddings_list as value as json to system to mimic persistence
#             job_data_file = f"./database/job_postings_{unique_id}.json"
#             job_page_dict = {unique_id: file_embedding}
#             # print(f"job page dict: {job_page_dict}")
#             with open(job_data_file, "w", encoding="utf-8") as f:
#                 json.dump(job_page_dict, f)
        
#         # Return PDF embeddings_list: List[List[List[float]]] and session vector store for entire research session
#         return {"embeddings_list": embeddings_list, "session_vector_store": get_session_vector_store(session_id)}
#     except HttpError as error:
#         print(f"An error occurred: {error}")
#     print("\n----------------------------------------\n")

async def embed_website(inputs):
    """Extension of run_google_searches to download PDFs and embed them."""
    print(f"embed_website inputs: {inputs}")
    embedding_method = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), show_progress_bar=True, chunk_size=100, skip_empty=True)
    
    id = inputs['id'] # job_id or profile_id
    image_path = ""
    name = ""
    url = inputs['url']
    type = inputs['type']
    if type == "job":
        session_id = random_base36_string()
        set_current_session_id(session_id)
        file_embedder = FileEmbedder(session_id)
    elif type == "profile":
        session_id = inputs['session_id'] # job_id
        print(f"profile session_id: {session_id}")
        file_embedder = FileEmbedder(session_id, pre_delete_collection=False)
        image_path = inputs['image_path']
        name = inputs['name']
    
    ### Initialize text splitter
    text_splitter = SemanticChunker(embeddings=embedding_method)
    
    try:
        embeddings_list = []
        
        ### Use unique id which was generated from React side which will be used to fetch embeddings for url
        unique_id = id
        upload_folder = Path("./uploaded_files")
        
        ### Vector Embedding
        # Embed website to session vector store
        file_extension = "html"
        file_embedding = None
        try:
            glob_pattern = f"**/*.{file_extension}"
            if type == "job":
                file_embedding = await file_embedder.process_and_embed_file(directory_path=upload_folder, file_path=None, unique_id=unique_id, url=url, embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter)
            elif type == "profile":
                new_driver = get_driver(session_id)
                # print(f"fetched driver: {new_driver}")
                file_embedding = await file_embedder.process_and_embed_file(driver=new_driver, directory_path=upload_folder, file_path=image_path, unique_id=unique_id, url=url, embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter, name=name)
        except Exception as e:
            raise f"Error processing url {url}: {e}"
        
        # Append to embeddings
        embeddings_list.append(file_embedding)
        
        # IFF type == job
        if type == "job":
            # Save the dictionary that uses unique_id as key and embeddings_list as value as json to system to mimic persistence
            job_data_file = f"./database/job_postings_{unique_id}.json"
            job_page_dict = {unique_id: file_embedding}
            async with aiofiles.open(job_data_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(job_page_dict))
        
        # Return PDF embeddings_list: List[List[List[float]]] and session vector store for entire research session
        return {"embeddings_list": embeddings_list, "session_vector_store": get_session_vector_store(session_id, False)}
    except HttpError as error:
        print(f"An error occurred: {error}")
    print("\n----------------------------------------\n")
    
# def call_populated_function(llm_response, id):
#     """
#     Checks if a responsed called a tool (funtion), apply this tool and return the response.
#     """
#     escalator = Escalator()
    
#     print("Call populated function called!")
#     try:
#         # print(f"llm response type: {type(llm_response)}")
#         # print(f"llm response: {llm_response}")
#         # tool_calls = llm_response.get('additional_kwargs', {}).get('tool_calls', [])
#         # Extract the additional kwargs
#         additional_kwargs = llm_response.additional_kwargs if hasattr(llm_response, 'additional_kwargs') else {}
        
#         # Extract tool calls from the additional kwargs
#         tool_calls = additional_kwargs.get('tool_calls', [])
#         if tool_calls:
#             # Assuming there's only one tool call per message for simplicity
#             tool_call = tool_calls[0]
#             if tool_call['function']['name'] == "extract_relevant_search_tags":
#                 arguments = json.loads(tool_call['function']['arguments'])
#                 location = arguments["location"]
#                 current_company = arguments["current_company"]
#                 industry = arguments["industry"]
#                 print(f"linkedin search filters: {[location, current_company, industry]}")
#                 linkedin_search_tags = escalator.extract_relevant_search_tags(id=id, location=location, current_company=current_company, industry=industry)
#                 # {"content": llm_response.choices[0].message.content, "internet_search": False}
#                 print(f"{{'tags': {linkedin_search_tags}}}")
#                 return linkedin_search_tags
#         else:
#             return llm_response

#     except Exception as e:
#         print(f"An error occurred: {e}")

###
# def call_populated_function(wrapped_response):
#     escalator = Escalator()
#     llm_response = wrapped_response.llm_response
#     additional_data = wrapped_response.additional_data
#     print(f"additional data: {additional_data}")
#     try:
#         additional_kwargs = llm_response.additional_kwargs if hasattr(llm_response, 'additional_kwargs') else {}
#         tool_calls = additional_kwargs.get('tool_calls', [])
#         if tool_calls:
#             tool_call = tool_calls[0]
#             if tool_call['function']['name'] == "extract_relevant_search_tags":
#                 arguments = json.loads(tool_call['function']['arguments'])
#                 location = arguments["location"]
#                 current_company = arguments["current_company"]
#                 industry = arguments["industry"]
#                 linkedin_search_tags = escalator.extract_relevant_search_tags(
#                     id=additional_data['id'], location=location, current_company=current_company, industry=industry
#                 )
#                 additional_data['linkedin_search_tags'] = linkedin_search_tags
#                 return WrappedResponse(llm_response=llm_response, additional_data=additional_data)
#         return wrapped_response
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None 
###
        
class WrappedResponse:
    def __init__(self, llm_response, additional_data):
        self.llm_response = llm_response
        self.additional_data = additional_data

def generate_prompt(pipeline_data, type):
    # Select the appropriate template based on the type
    if type == "job":
        template = WEB_SCRAPING_TEMPLATE
    elif type == "profile":
        template = PROFILE_SCRAPING_TEMPLATE
    else:
        raise ValueError("Invalid type. Must be either 'job' or 'profile'.")

    # print(f"chosen template: {template}")
    
    prompt_template = ChatPromptTemplate.from_template(template=template)
    prompt_output = prompt_template.format(**pipeline_data)
    if type == "profile":
        print(f"pipeline data for profile: {pipeline_data}")
    # Wrap the prompt output in the expected message format
    human_message = HumanMessage(content=prompt_output)
    print("Generate Prompt Successful!")
    return {
        "prompt_output": [human_message],  # LLM expects a list of messages
        "context": pipeline_data,  # Preserve the original context
        "type": type
    }

# escalator_global = Escalator()
# def call_populated_function(wrapped_response):
#     llm_response = wrapped_response.llm_response
#     additional_data = wrapped_response.additional_data
    
#     try:
#         additional_kwargs = llm_response.additional_kwargs if hasattr(llm_response, 'additional_kwargs') else {}
#         tool_calls = additional_kwargs.get('tool_calls', [])
#         if tool_calls:
#             tool_call = tool_calls[0]
#             if tool_call['function']['name'] == "extract_relevant_search_tags":
#                 arguments = json.loads(tool_call['function']['arguments'])
#                 location = arguments["location"]
#                 current_company = arguments["current_company"]
#                 title = arguments["title"]
#                 team = arguments["team"]
#                 linkedin_search_tags = escalator_global.extract_relevant_search_tags(
#                     id=additional_data['id'], location=location, current_company=current_company, title=title, team=team
#                 )
#                 additional_data['linkedin_search_tags'] = linkedin_search_tags
                
#                 return linkedin_search_tags
#             elif tool_call['function']['name'] == "extract_relevant_profile_keywords":
#                 escalator = Escalator()
#                 arguments = json.loads(tool_call['function']['arguments'])
#                 print(f"arguments for extract_relevant_profile_keywords: {arguments}, url: {additional_data['url']}, image path: {additional_data['image_path']}")
#                 name = arguments["name"]
#                 current_company = arguments["current_company"]
#                 title = arguments["title"]
#                 team = arguments["team"]
#                 most_recent_school = arguments["most_recent_school"]
#                 undergraduate_school = arguments["undergraduate_school"]
#                 total_years_employed = arguments["total_years_employed"]
                
#                 profile_keywords = escalator.extract_relevant_profile_keywords(
#                     url=additional_data['url'], image_path=additional_data['image_path'], id=additional_data['id'], name=name, current_company=current_company, title=title, team=team, most_recent_school=most_recent_school, undergraduate_school=undergraduate_school, total_years_employed=total_years_employed
#                 )
#                 additional_data['profile_keywords'] = profile_keywords
                
#                 return profile_keywords
#         return wrapped_response
#     except Exception as e:
#         print(f"An error occurred in call_populated_function: {e}")
#         return None

def wrap_prompt_and_llm(pipeline_data, llm_model):
    print(f"pipeline data: {pipeline_data}")
    # Extract the context before invoking the template
    context = {
        "id": pipeline_data["id"],
        "instruction": pipeline_data["instruction"],
        "file_embedding_keys": pipeline_data["file_embedding_keys"],
        "context": pipeline_data["context"]
    }

    # Invoke the ChatPromptTemplate
    prompt_output = (pipeline_data)

    # Prepare the input for the LLM with the prompt output
    # llm_input = {
    #     "messages": prompt_output,
    #     "context": context
    # }

    # Invoke the LLM
    print(f"What the fuck?: {ChatPromptTemplate.from_template(template=WEB_SCRAPING_TEMPLATE)}")
    llm_response = ChatPromptTemplate.from_template(template=WEB_SCRAPING_TEMPLATE) | llm_model(pipeline_data)

    # Wrap the response with the original context
    wrapped_response = WrappedResponse(llm_response=llm_response, additional_data=context)
    return wrapped_response
###

# Correctly setting up partial functions
custom_question_getter = partial(get_question)
question_context_retriever_runnable = RunnableLambda(lambda question_runnable_output: retrieve_context_from_question(question_runnable_output))
custom_context_getter = partial(get_context_and_trace, key="context") # change to context
custom_augment_getter = partial(get_augment_and_trace)
redirect_test_runnable = partial(redirect_test, key="question")
process_response_runnable = partial(process_response, key='response')

# jupyter_solver = JupyterSolver()

# preprocess_chain is a sequential(not parallel) chain that does the following:
# 1. Passes the context, augment, and file_embedding_keys to the chat completions api and receives a json that identifies the function that must be called with arguments. This function is process_files_and_generate_response of the JupyterSolver Class
# 2. Calls process_files_and_generate_response with proper arguments that were returned by the chat completions api.
# 3. Once the document(s) that hold the questions are identified through the first chain that uses the chat completions api, solve_chain will pipe that to another chain that this time uses the chat completions api to
# extract a list of problem strings from the document(s) that hold the questions.
# 4. Generate a jupyter notebook where each cell correspond to each extracted problem from the document augment.
# preprocess_chain = (
#     # This chain needs to be modified so that the final return value is the json that specifies the function that needs to be called with arguments.
#     (RunnableParallel(
#         context=(itemgetter("question") | ordered_compression_default_retriever),
#         augment=custom_augment_getter,
#         question=redirect_test_runnable,
#         file_embedding_keys=itemgetter("file_embedding_keys")
#     ) |
#     RunnableParallel(
#         response=(ChatPromptTemplate.from_template(template=PREPROCESSING_TEMPLATE) | llm_jupyter_preprocessor),
#         docs=custom_context_getter,
#         augment=lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
#         augments=itemgetter("augment")
#     )) |
#     process_response_runnable |
#     RunnableParallel(
#         problem_extraction=lambda inputs: jupyter_solver.solve_problems(problems=inputs[0], problem_file_path=inputs[1], pointer_file_path=inputs[2])
#     )
# ).with_types(input_type=RagInput)

fetch_context = partial(fetch_context_document_with_score)
# Google Research
# run_searches_and_embed_results = partial(run_google_searches_and_embed_pdfs)
# return_converted_docs = RunnableLambda(lambda run_searches_and_embed_results_output: convert_embeddings_to_docs(**run_searches_and_embed_results_output)) # Expects embeddings_list and session_vector_store as arguments

# Chain for returning to the user three most relevant academic paper given their query and explaining what they are about
# research_chain = (
#     RunnableParallel(
#         # subchain that normally returns context but returns top 3 pdfs if they exist
#         context=(run_searches_and_embed_results | return_converted_docs),
#         question=itemgetter("question"),
#         file_embedding_keys=itemgetter("file_embedding_keys") # Optional, but needs to be here in order for augment_context_with_file_embeddings to reset session vector store
#     ) |
#     RunnableParallel(
#         answer=(ChatPromptTemplate.from_template(template=RESEARCH_TEMPLATE) | llm_conversation),
#         augment = lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
#     )
# ).with_types(input_type=RagInput)

### Escalator Chains
# scrape_and_embed_url = partial(embed_website)
# return_converted_docs = RunnableLambda(lambda scrape_and_embed_url_output: convert_embeddings_to_docs(**scrape_and_embed_url_output))
# wrap_llm_response = RunnableLambda(lambda pipeline_data: wrap_llm_with_context(pipeline_data, llm_linkedin_search_tags_extractor))
# call_extract_relevant_search_tags = RunnableLambda(lambda llm_linkedin_search_tags_extractor_response: call_populated_function(**llm_linkedin_search_tags_extractor_response))

# Define the chain with RunnableParallel and RunnableLambda
# scrape_and_embed_url = partial(embed_website)
# return_converted_docs = RunnableLambda(lambda data: convert_embeddings_to_docs(**data))
# wrap_llm_response = RunnableLambda(lambda pipeline_data: wrap_llm_with_context(pipeline_data, llm_linkedin_search_tags_extractor))
# call_extract_relevant_search_tags = RunnableLambda(lambda wrapped_response: call_populated_function(wrapped_response))

# Escalator Web Scraping Chain
# web_scraping_chain = (
#     RunnableParallel(
#         context=(scrape_and_embed_url | return_converted_docs),
#         instruction=itemgetter("instruction"),
#         id=itemgetter("id"),
#         file_embedding_keys=itemgetter("file_embedding_keys")
#     ) |
#     RunnableParallel(
#         result=(ChatPromptTemplate.from_template(template=WEB_SCRAPING_TEMPLATE) | wrap_llm_response | call_extract_relevant_search_tags),
#         augment = lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
#     )
# ).with_types(input_type=LinkedinInput)

# Wrapping both the template and the LLM
# wrap_prompt_and_llm_response = RunnableLambda(lambda pipeline_data: wrap_prompt_and_llm(
#     pipeline_data, 
#     llm_linkedin_search_tags_extractor
# ))

# Wrapping both the template and the LLM
# generate_prompt_runnable = RunnableLambda(lambda pipeline_data: generate_prompt(pipeline_data, pipeline_data['type']))
# call_llm_with_context_runnable = RunnableLambda(lambda prompt_data: call_llm_with_context(prompt_data))
# call_extract_relevant_search_tags = RunnableLambda(lambda wrapped_response: call_populated_function(wrapped_response))

# web_scraping_chain = (
#     RunnableParallel(
#         context=(scrape_and_embed_url | return_converted_docs),
#         instruction=itemgetter("instruction"),
#         id=itemgetter("id"),
#         type=itemgetter("type"),
#         file_embedding_keys=itemgetter("file_embedding_keys")
#     ) |
#     RunnableParallel(
#         result=(generate_prompt_runnable | call_llm_with_context_runnable | call_extract_relevant_search_tags),
#         augment=lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
#     )
# ).with_types(input_type=dict)

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

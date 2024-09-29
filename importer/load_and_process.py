from concurrent.futures import ProcessPoolExecutor, as_completed

# import requests
import base64
import bs4
import os
import shutil
import sys
from typing import List, Optional
import uuid
import time
import pdfkit
import tempfile
import weasyprint
import json
# print(f"sys path: {sys.path}")
# Add the parent directory to sys.path to allow for correct module resolution
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import asyncio
import aiofiles
import aiofiles.os as aios
import aioshutil
import aiohttp
from pyppeteer import launch

import nbformat
import fitz # PyMuPDF for PDF processing
import pytesseract  # Assuming you choose pytesseract for OCR
from langchain_community.document_transformers.embeddings_redundant_filter import _DocumentWithState

from dotenv import load_dotenv
# pdf, ipynb, txt
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader, TextLoader, NotebookLoader, UnstructuredCSVLoader, UnstructuredImageLoader, WebBaseLoader, SeleniumURLLoader
from langchain_core.documents.base import Document
# Code
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores.pgvector import PGVector

from langchain.vectorstores.deeplake import DeepLake
from langchain.text_splitter import Language
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI
from nbconvert import MarkdownExporter
from PIL import Image
from unstructured.partition.auto import partition

from pathlib import Path

from requests_html import AsyncHTMLSession

from io import BytesIO

import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait

from app.embeddings_manager import get_session_vector_store, store_embedding, get_current_session_id, store_pdf_path, get_window_handler
from app.config import EMBEDDING_MODEL, PG_COLLECTION_NAME

load_dotenv()

client = OpenAI()

chatgpt_base = ChatOpenAI(temperature=0.7, model='gpt-4o', max_tokens=4096, streaming=True)

def custom_embed_entities(client: OpenAI, entities, model="text-embedding-ada-002"):
    """
    Embed a list of documents using OpenAI's API directly, bypassing tiktoken.
    :param documents: A list of strings (documents) to be embedded.
    :param openai_api_key: Your OpenAI API key.
    :param model: The model to use for embeddings.
    :return: A list of embeddings.
    """
    embeddings = []
    for entity in entities:
        try:
            response = client.embeddings.create(input=entity, model=model)
            embeddings.append(response['data'][0]['embedding'])
        except Exception as e:
            print(f"Error embedding document: {e}")
            embeddings.append(None)  # Or handle as appropriate
    return embeddings

# Function to encode images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function for image summaries
async def summarize_image(index: int, encoded_image):
    print("Summarizing Image using gpt-4o")
    prompt = [
        AIMessage(content="You are a GPT that analyzes and describes images."),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the contents of this image in complete detail."},
            {"type": "text", "text": f"Before each image description, include Figure {index + 1}:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = chatgpt_base.invoke(prompt)
    return response.content

async def summarize_images_job(image_elements):
    # Create a list of tasks using asyncio.create_task
    tasks = [asyncio.create_task(summarize_image(i, ie)) for i, ie in enumerate(image_elements)]
    # Gather all the results
    return await asyncio.gather(*tasks)

class jupyter_notebook_creator:
    def extract_questions(self):
        """Method that utilizes the chat completions api to extract questions that need to be solved from current list of passed files."""
        # This method will be called from server.py and we will first test if it can list all currently passed files correctly.
        
    # def create_notebook_from_questions(questions):
    #     nb = nbformat.v4.new_notebook()
    #     for question in questions:
    #         # Generate markdown cell for the question
    #         question_cell = nbformat.v4.new_markdown_cell(f"### Question:\n{question}")
    #         nb.cells.append(question_cell)
            
    #         # Use chat completions API to get an answer or code snippet
    #         answer = get_chat_completion(question)
            
    #         # Determine if the answer should be a code cell or markdown cell
    #         if is_code_snippet(answer):
    #             answer_cell = nbformat.v4.new_code_cell(answer)
    #         else:
    #             answer_cell = nbformat.v4.new_markdown_cell(answer)
            
    #         nb.cells.append(answer_cell)
        
    #     # Save the new notebook
    #     with open("output_notebook.ipynb", "w", encoding="utf-8") as f:
    #         nbformat.write(nb, f)

class InitialProcessor:
    def process_image(self, file_path):
        # Convert the image file to text using OCR
        text = pytesseract.image_to_string(Image.open(file_path))
        return text
    
    def process_ipynb(self, file_path):
        """Method that reads a jupyter notebook cell by cell and passes each one individually to an LLM API endpoint"""
        with open(file_path, encoding='utf8') as f:
            nb = nbformat.read(f, as_version=4)
        exporter = MarkdownExporter()
        body, _ = exporter.from_notebook_node(nb)
        return body
    
    # def process_pdf(self, file_path):
    #     text = ""
    #     with fitz.open(file_path) as doc:
    #         for page in doc:
    #             text += page.get_text()
    #     return text

    # def process_text_file(self, file_path):
    #     with open(file_path, 'r') as file:
    #         text = file.read()
    #     return text
    
class WebProcessor:
    driver = None
    wait = None
    # https://chromedevtools.github.io/devtools-protocol/tot/Page#method-printToPDF
    print_options = {
        'landscape': False,
        'displayHeaderFooter': False,
        'printBackground': True,
        'preferCSSPageSize': True,
        # 'paperWidth': 6.97,
        # 'paperHeight': 16.5,
        'paperWidth': 8.3,
        'paperHeight': 11.7,
    }

    def __init__(self):
        self.session = AsyncHTMLSession()
    
    async def download_url_as_pdf(self, is_profile: bool, url, upload_folder: Path, session_id=None, file_path=None, name=None) -> Path:
        window = None
        if is_profile:
            window = get_window_handler(url)
            print(f"get window handler! url: {url} window: {window}")
            self.driver.switch_to.window(window)
            print(f"switched to window handler: {self.driver.current_window_handle}")
            # On Purpose, without this we'll get too many requests error
            time.sleep(1.5)
        else:
            print(f"download url as pdf called with url: {url}")
            self.driver.get(url)

            # Wait until the document is fully loaded; no async, we want to at least save the pdf before telling the driver to do the next task
            self.wait_for_page_load()
        
        print_options = self.print_options.copy()
        print(f"self.driver window handler right before print to pdf: {self.driver.current_window_handle} with url: {url}")
        result = self._send_devtools(self.driver, "Page.printToPDF", print_options)
        binary_pdf = base64.b64decode(result['data'])
        
        # Up until here, each driver operations must be sequential
        
        file = BytesIO()
        file.write(binary_pdf)
        file.seek(0)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            temp_pdf_path = tmp_pdf.name
            
        # print(f"download url as pdf middle test: {url}")
        print(f"session_id: {session_id}, image path: {file_path}")
        # if session_id != "" and file_path != None and name != "":
        #     global_people_list = global_people.get(session_id)
        #     if global_people_list:
        #         global_people_list.append({name : file_path})
        #         global_people[session_id] = global_people_list # What should go here?
        #     else:
        #         global_people_list = []
        #         global_people_list.append({name : file_path})
        #         global_people[session_id] = global_people_list
        #     print(f"global people: {global_people[session_id]}")

        async with aiofiles.open(temp_pdf_path, 'wb') as f:
            await f.write(file.read())
            
        file_name = os.path.splitext(os.path.basename(temp_pdf_path))[0]
        
        pdf_parent_path = upload_folder / f"{file_name}"
        
        # aios == aiofiles.os
        await aios.makedirs(pdf_parent_path, exist_ok=True)
        
        pdf_path = pdf_parent_path / f"{os.path.splitext(os.path.basename(temp_pdf_path))[0]}.pdf"

        await aios.rename(temp_pdf_path, pdf_path)

        print(f"Saved website PDF to: {str(pdf_path)} extracted from url: {url} using window handler: {window}")

        return pdf_path, pdf_parent_path
    
    @staticmethod
    def _send_devtools(driver, cmd, params):
        """
        Works only with chromedriver.
        Method uses cromedriver's api to pass various commands to it.
        """
        resource = "/session/%s/chromium/send_command_and_get_result" % driver.session_id
        url = driver.command_executor._url + resource
        body = json.dumps({'cmd': cmd, 'params': params})
        print(f"_send_devtools print url: {url}")
        response = driver.command_executor._request('POST', url, body)
        return response.get('value')
    
    # @staticmethod
    # async def _send_devtools(driver, cmd, params):
    #     """
    #     Works only with chromedriver.
    #     Method uses chromedriver's api to pass various commands to it.
    #     """
    #     resource = "/session/%s/chromium/send_command_and_get_result" % driver.session_id
    #     url = driver.command_executor._url + resource
    #     body = json.dumps({'cmd': cmd, 'params': params})

    #     async with aiohttp.ClientSession() as session:
    #         async with session.post(url, data=body) as response:
    #             response_data = await response.json()
    #             if 'value' in response_data:
    #                 return response_data['value']
    #             else:
    #                 raise KeyError(f"Unexpected response format: {response_data}")
                
    async def await_for_page_load(self):
        """Waits for the page to fully load by checking the document's readyState."""
        while True:
            await asyncio.sleep(1)
            ready_state = self.driver.execute_script('return document.readyState;')
            if ready_state == 'complete':
                break
            
    def wait_for_page_load(self):
        """Waits for the page to fully load by checking the document's readyState."""
        while True:
            time.sleep(5)
            ready_state = self.driver.execute_script('return document.readyState;')
            if ready_state == 'complete':
                break
    
class FileProcessor:
    async def process_documents(self, directory_path: Path, file_path, glob_pattern, loader_cls, text_splitter, embeddings, session_vector_store=None, session_id=None):
        """To ensure that the process_documents method accurately processes only PDF files, one can modify the glob_pattern parameter used in the DirectoryLoader to specifically target PDF files. This adjustment will make the method more focused and prevent it from attempting to process files of other types, which might not be suitable for the intended processing pipeline."""
        print(f"loader_cls: {loader_cls}")
        directory_path_str = str(directory_path)
        
        # pdf_glob_pattern = "**/*.pdf"  # Updated to specifically target PDF files
        extracted_extension = glob_pattern.split('.')[-1]
        
        # Read PDF using PDF Loader (Only Text)
        loader = DirectoryLoader(
            os.path.abspath(directory_path_str),
            glob=glob_pattern,  # Use the PDF-specific glob pattern
            use_multithreading=True,
            show_progress=True,
            max_concurrency=50,
            loader_cls=loader_cls,
        )
        docs = loader.load()
        
        # Get the base name (without extension) of each pdf file in the directory_path
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        chunks = docs
        
        print(f"docs after chunking: {chunks}")
        store, current_embeddings = await PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            chunk_size=10,
            collection_name=PG_COLLECTION_NAME,
            connection_string=os.getenv("POSTGRES_URL"),
            pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
        )
        
        if session_vector_store:
            # print(f"docs after chunking: {chunks}")
            store, current_embeddings = await session_vector_store.from_documents(
                documents=chunks,
                embedding=embeddings,
                chunk_size=100,
                collection_name=session_id,
                connection_string=os.getenv("POSTGRES_URL"),
                pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
            )
        return current_embeddings

    async def process_code(self, repo_path, suffixes, language_setting, embeddings, parser_threshold=500, session_vector_store=None, session_id=None):
        print("Began Processing Code")
        # Configure the loader with appropriate settings
        loader = GenericLoader.from_filesystem(
            path=repo_path,
            glob="**/*",
            suffixes=suffixes,
            exclude=[f"**/non-utf8-encoding{suffixes[0]}"],  # Exclude all files that are non-utf8-encoding
            show_progress=True,
            parser=LanguageParser(language=language_setting, parser_threshold=parser_threshold),
        )

        documents = loader.load()

        # Split code into meaningful chunks
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language_setting, chunk_size=100, chunk_overlap=10
        )
        chunks = code_splitter.split_documents(documents)

        store, current_embeddings = await PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            chunk_size=100,
            collection_name=PG_COLLECTION_NAME,
            connection_string=os.getenv("POSTGRES_URL"),
            pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
        )
        if session_vector_store:
            store, current_embeddings = await session_vector_store.from_documents(
                documents=chunks,
                embedding=embeddings,
                chunk_size=100,
                collection_name=session_id,
                connection_string=os.getenv("POSTGRES_URL"),
                pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
            )
        print(f"chunks vectorized: {len(chunks)}")
        return current_embeddings
    
    async def process_images_as_text(self, directory_path, file_path, glob_pattern, loader_cls, text_splitter, embeddings, session_vector_store=None, session_id=None):
        """To ensure that the process_documents method accurately processes only PDF files, one can modify the glob_pattern parameter used in the DirectoryLoader to specifically target PDF files. This adjustment will make the method more focused and prevent it from attempting to process files of other types, which might not be suitable for the intended processing pipeline."""
        # print(f"image file_path: {file_path}")
        
        # Read image from directory_path and encode it        
        encoded_image = encode_image(file_path)
        
        # Image to string summary
        image_summary = await summarize_image(0, encoded_image)
        # print(f"summarized image string: {image_summary}")
        
        # Generate txt file from image_summary string
        image_txt_path = os.path.splitext(file_path)[0] + ".txt" # e.g. ./uploaded_files/darth_vader.txt
        # print(f"image_txt_path: {image_txt_path}")
        with open(image_txt_path, "w", encoding="utf-8") as text_file:
            text_file.write(image_summary)
        
        loader = DirectoryLoader(
            os.path.abspath(directory_path),
            glob="**/*.txt", # Manually set to txt
            use_multithreading=True,
            show_progress=True,
            max_concurrency=50,
            loader_cls=loader_cls, # txt loader
        )
        docs = loader.load()
        # Split summarized List[Document] of an image into more meaningful chunks, which is also a List[Document]
        chunks = text_splitter.split_documents(docs)
        
        # Store in vector store!
        store, current_embeddings = await PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            chunk_size=100,
            collection_name=PG_COLLECTION_NAME,
            connection_string=os.getenv("POSTGRES_URL"),
            pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
        )
        
        if session_vector_store:
            print("Storing to session vector store!")
            store, current_embeddings = await session_vector_store.from_documents(
                documents=chunks,
                embedding=embeddings,
                chunk_size=100,
                collection_name=session_id,
                connection_string=os.getenv("POSTGRES_URL"),
                pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
            )
        print(f"chunks vectorized: {len(chunks)}")
        return current_embeddings

class FileEmbedder:
    def __init__(self, session_id, pre_delete_collection=True):
        self.session_vector_store = get_session_vector_store(session_id, pre_delete_collection) # pre_delete_collection = True by default
        self.session_id = session_id
    
    # async def process_and_embed_file(self, unique_id, directory_path, file_path, embeddings, glob_pattern=None, suffixes=None, language_setting=None, text_splitter=None, url=None, driver=None) -> Optional[List[List[float]]]:
    #     # Initialize FileProcessor to access its processing methods
    #     file_processor = FileProcessor()
        
    #     # Mapping of file extensions to processing functions and their arguments
    #     process_map = {
    #         'pdf': {"func": file_processor.process_documents, "args": {"loader_cls": UnstructuredPDFLoader}},
    #         'ipynb': {"func": file_processor.process_documents, "args": {"loader_cls": NotebookLoader}},
    #         'txt': {"func": file_processor.process_documents, "args": {"loader_cls": TextLoader}},
    #         'csv': {"func": file_processor.process_documents, "args": {"loader_cls": UnstructuredCSVLoader}},
    #         'png': {"func": file_processor.process_images_as_text, "args": {"loader_cls": TextLoader}},
    #         'html': {"func": file_processor.process_website, "args": {"loader_cls": SeleniumURLLoader}},
    #         'code': {"func": file_processor.process_code, "args": {"suffixes": suffixes, "language_setting": language_setting}},
    #         # Image processing to be implemented later
    #     }
        
    #     # Determine the file type from the glob_pattern or suffixes
    #     file_type = "code" if suffixes else glob_pattern.split(".")[-1]
        
    #     # Fetch the processing function and its specific arguments
    #     processing_info = process_map.get(file_type, None)
        
    #     if processing_info:
    #         process_func = processing_info["func"]
    #         process_args = processing_info["args"]

    #         if file_type in ['pdf', 'ipynb', 'txt', 'csv']:
    #             current_embeddings = await process_func(directory_path=directory_path, file_path=file_path, glob_pattern=glob_pattern, **process_args, text_splitter=text_splitter, embeddings=embeddings, session_vector_store=self.session_vector_store, session_id=self.session_id)
    #         elif file_type == 'code':
    #             current_embeddings = await process_func(repo_path=directory_path, **process_args, embeddings=embeddings, session_vector_store=self.session_vector_store, session_id=self.session_id)
    #         elif file_type == 'png':
    #             # Embed Images
    #             current_embeddings = await process_func(directory_path=str(directory_path), file_path=file_path, glob_pattern=glob_pattern, **process_args, text_splitter=text_splitter, embeddings=embeddings, session_vector_store=self.session_vector_store, session_id=self.session_id)
    #         elif file_type == 'html':
    #             # Embed Website
    #             current_embeddings = await process_func(driver=driver, unique_id=unique_id, upload_folder=directory_path, url=url, **process_args, text_splitter=text_splitter, embeddings=embeddings, session_vector_store=self.session_vector_store, session_id=self.session_id)
    #         else:
    #             # Handling for not yet implemented or unrecognized file types
    #             return {"error": f"Processing for file type {file_type} is not implemented or unrecognized."}
            
    #         # Assuming 'current_embeddings' are returned correctly from processing functions
    #         if current_embeddings:
    #             print(f"File with file_embedding_key: {unique_id}")
    #             store_embedding(unique_id, current_embeddings)
    #         return current_embeddings
    #     else:
    #         return {"error": f"No processing function found for file type: {file_type}"}
    
    async def process_and_embed_file(self, unique_id, directory_path, file_path, embeddings, glob_pattern=None, suffixes=None, language_setting=None, text_splitter=None, url=None, driver=None, name=None) -> Optional[List[List[float]]]:
        # Initialize FileProcessor to access its processing methods
        file_processor = FileProcessor()
        
        # Mapping of file extensions to processing functions and their arguments
        process_map = {
            'pdf': {"func": file_processor.process_documents, "args": {"loader_cls": UnstructuredPDFLoader}},
            'ipynb': {"func": file_processor.process_documents, "args": {"loader_cls": NotebookLoader}},
            'txt': {"func": file_processor.process_documents, "args": {"loader_cls": TextLoader}},
            'csv': {"func": file_processor.process_documents, "args": {"loader_cls": UnstructuredCSVLoader}},
            'png': {"func": file_processor.process_images_as_text, "args": {"loader_cls": TextLoader}},
            'code': {"func": file_processor.process_code, "args": {"suffixes": suffixes, "language_setting": language_setting}},
        }
        
        # Determine the file type from the glob_pattern or suffixes
        file_type = "code" if suffixes else glob_pattern.split(".")[-1]
        
        # Fetch the processing function and its specific arguments
        processing_info = process_map.get(file_type, None)
        
        if processing_info:
            process_func = processing_info["func"]
            process_args = processing_info["args"]

            if file_type in ['pdf', 'ipynb', 'txt', 'csv']:
                current_embeddings = await process_func(directory_path=directory_path, file_path=file_path, glob_pattern=glob_pattern, **process_args, text_splitter=text_splitter, embeddings=embeddings, session_vector_store=self.session_vector_store, session_id=self.session_id)
            elif file_type == 'code':
                current_embeddings = await process_func(repo_path=directory_path, **process_args, embeddings=embeddings, session_vector_store=self.session_vector_store, session_id=self.session_id)
            elif file_type == 'png':
                # Embed Images
                current_embeddings = await process_func(directory_path=str(directory_path), file_path=file_path, glob_pattern=glob_pattern, **process_args, text_splitter=text_splitter, embeddings=embeddings, session_vector_store=self.session_vector_store, session_id=self.session_id)
            elif file_type == 'html':
                # Embed Website
                current_embeddings = await process_func(driver=driver, unique_id=unique_id, upload_folder=directory_path, url=url, **process_args, text_splitter=text_splitter, embeddings=embeddings, session_vector_store=self.session_vector_store, session_id=self.session_id, file_path=file_path, name=name)
            else:
                return {"error": f"Processing for file type {file_type} is not implemented or unrecognized."}
            
            if current_embeddings:
                print(f"File with file_embedding_key: {unique_id}")
                store_embedding(unique_id, current_embeddings)
            return current_embeddings
        else:
            return {"error": f"No processing function found for file type: {file_type}"}

class RepositoryProcessor:
    def __init__(self, clone_path, deeplake_path, embedding_method, allowed_extensions=None):
        if allowed_extensions is None:
            allowed_extensions = ['.py', '.ipynb', '.md']
        self.clone_path = clone_path
        self.deeplake_path = deeplake_path
        self.allowed_extensions = allowed_extensions
        self.docs = []
        self.texts = []
        self.model_name = EMBEDDING_MODEL
        # "sentence-transformers/all-MiniLM-L6-v2"
        self.model_kwargs = {"model_name": self.model_name}
        self.hf = embedding_method
        # HuggingFaceEmbeddings(**self.model_kwargs)

    def extract_all_files(self):
        for dirpath, dirnames, filenames in os.walk(self.clone_path):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in self.allowed_extensions:
                    try:
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e:
                        print(f"Failed to process file {file}: {e}")

    def chunk_files(self):
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)

    def embed_and_store(self):
        db = DeepLake(dataset_path=self.deeplake_path, embedding_function=self.hf)
        db.add_documents(self.texts)
        print(f"Embedded and stored {len(self.texts)} documents.")

    def process_repository(self):
        self.extract_all_files()
        self.chunk_files()
        self.embed_and_store()
        
    def delete_embeddings(self):
        db = DeepLake(dataset_path=self.deeplake_path, embedding_function=self.hf)
        db.delete(delete_all=True, large_ok=True)
        print(f"Deleted documents in {self.deeplake_path}.")

async def main():
    # Initialize the embeddings and text splitter
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), skip_empty=True, chunk_size=100)
    text_splitter = SemanticChunker(embeddings=embeddings)
    file_processor = FileProcessor()

    # Process PDFs
    # await file_processor.process_documents("./source_docs", "**/*.pdf", UnstructuredPDFLoader, text_splitter, embeddings)

    # Process Code Files (main languages Using the language parser itself)
    
    # Self Repository
    # await file_processor.process_code("./frontend", [".py"], Language.PYTHON, embeddings)
    # await file_processor.process_code("./frontend", [".js"], Language.JS, embeddings)
    # await file_processor.process_code("./app", [".py"], Language.PYTHON, embeddings)
    # await file_processor.process_code("./importer", [".py"], Language.PYTHON, embeddings)
    
    # venv
    # await file_processor.process_code("./venv/lib/python3.11/site-packages/langserve", [".py"], Language.PYTHON, embeddings)
    # await file_processor.process_code("./venv/lib/python3.11/site-packages/langserve", [".js"], Language.JS, embeddings)
    
    # source code store
    # await file_processor.process_code("./source_code", [".py"], Language.PYTHON, embeddings)
    # await file_processor.process_code("./source_code", [".js"], Language.JS, embeddings)
    # await file_processor.process_code("./source_code", [".ts"], Language.TS, embeddings)

    # DeepLake: Repository Processor for code in multiple languages (after conversion of code to text)
    # repo_processor = RepositoryProcessor(clone_path="./", deeplake_path=os.path.expanduser("~/deeplake_data"), allowed_extensions=[".py", ".js", ".md", ".ts", ".tsx", ".html", ".css"])
    # print(os.path.expanduser("~/deeplake_data"))
    # repo_processor.process_repository()
    # repo_processor.delete_embeddings()

if __name__ == "__main__":
    asyncio.run(main())

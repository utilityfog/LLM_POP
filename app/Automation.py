import uuid
import httpx
import nbformat
import re

from fastapi import Path, Request
from fastapi.responses import RedirectResponse
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
from pathlib import Path as LibPath
from typing import Dict, List, Optional, Tuple
from nbclient import NotebookClient

from app.embeddings_manager import get_current_session_id, get_embedding_from_manager
from importer.load_and_process import FileEmbedder

class JupyterSolver:
    def convert_embeddings_to_documents(self, embeddings: List[List[float]], session_vector_store) -> List[List[Document]]:
        print("Converting received embeddings to documents")
        documents: List[List[Document]] = []
        if embeddings:
            for embedding in embeddings:
                summarized_embedding = session_vector_store.similarity_search_by_vector(embedding=embedding, k=1) # summarized_embedding is guaranteed to be a Document object
                documents.append(summarized_embedding)
        else:
            print("Current Embeddings None!")
            return
        # print(f"embeddings: {embeddings}")
        # print(f"documents: {documents}")
        return documents # type = List[List[Document]]
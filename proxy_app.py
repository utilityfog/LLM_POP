# proxy_app.py
from fastapi import FastAPI

proxy_app = FastAPI()

# Add your proxy middleware or routes if necessary
# For example:
from app.server import ASGIProxy

proxy_app.add_middleware(ASGIProxy)
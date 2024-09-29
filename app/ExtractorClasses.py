import os
from typing import List, Any

# from selenium import webdriver
import subprocess
import re
import undetected_chromedriver as uc
import chromedriver_autoinstaller

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import json

import time

import requests

from PIL import Image
from io import BytesIO

from app.embeddings_manager import get_file_path

global_location_names: {str : Any} = {}
global_locations: {str : Any} = {}

class Extractor:
    extract_relevant_api_arguments_json = [
        {
            "type": "function",
            "function": {
                "name": "extract_relevant_api_arguments",
                "description": ("Given a vector embedded user prompt that details where they want to travel to, extract these Tourpedia API arguments: "
                                "1. location"
                                "2. category"
                                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Physical location of where the user wants to travel to which is one of: (Berlin, Amsterdam, Barcelona, Dubai, London, Paris, Rome, Tuscany)"
                        },
                        "category": {
                            "type": "string",
                            "description": "location category which is one of: (attraction, restaurant, poi)"
                        }
                    },
                    "required": ["location", "category"]
                }
            }
        }
    ]
    
    extract_relevant_location_properties_json = [
        {
            "type": "function",
            "function": {
                "name": "extract_relevant_location_properties",
                "description": ("Given a vector embedded Location Review JSON, extract these location properties: "
                                "1. name"
                                "2. address"
                                "3. url"
                                "4. pros"
                                "5. cons"
                                "6. rating"
                                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the location"
                        },
                        "address": {
                            "type": "string",
                            "description": "Physical address of the location"
                        },
                        "url": {
                            "type": "string",
                            "description": "url of the location's details"
                        },
                        "pros": {
                            "type": "string",
                            "description": "Things that are good about the location"
                        },
                        "cons": {
                            "type": "string",
                            "description": "Things that are not good about the location"
                        },
                        "rating": {
                            "type": "integer",
                            "description": "Average rating given by visitors for the location"
                        }
                    },
                    "required": ["name", "address", "pros", "cons", "rating"]
                }
            }
        }
    ]
    
    def extract_relevant_api_arguments(self, location: str, category: str) -> List[str]:
        """Method called by an LLM once it extracts relevant Tourpedia arguments from its input"""
        print(f"extracted arguments: {[location, category]}")
        return [location, category]
    
    def extract_relevant_location_properties(self, id: str, name: str, address: str, url: str, pros: str, cons: str, rating: int) -> dict:
        """Method called by an LLM once it extracts relevant location properties from its input"""
        # Create a profile object
        attraction_object = {
            "id": id,
            "name": name,
            "address": address,
            "url": url,
            "pros": pros,
            "cons": cons,
            "rating": rating
        }
            
        location_txt_basename = get_file_path(id) # Saved location txt when the raw location object is vectorized
        print(f"location txt basename for {url}: {location_txt_basename} with id: {id}")
        
        return {location_txt_basename: attraction_object}
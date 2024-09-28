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

from app.embeddings_manager import get_pdf_path

global_profile_urls: {str : Any} = {}
global_people: {str : Any} = {}

class LinkedInProfileScraper:
    driver: uc.Chrome = None
        
    def get_chrome_version(self):
        """Gets the current installed Chrome version."""
        try:
            output = subprocess.check_output(['google-chrome', '--version'], stderr=subprocess.STDOUT).decode('utf-8')
            version = re.search(r'(\d+)\.(\d+)\.(\d+)\.(\d+)', output).group(0)
            return version
        except Exception as e:
            raise f"Failed to get Chrome version: {e}"
    
    def install_correct_chromedriver(self):
        """Install the correct chromedriver for the current Chrome version."""
        chromedriver_path = chromedriver_autoinstaller.install(True)
        return chromedriver_path
        
    def __init__(self, should_initialize_driver=True):
        if should_initialize_driver:
            chromedriver_path = self.install_correct_chromedriver()
            
            self.options = uc.ChromeOptions()
            self.options.add_argument('--headless')
            self.options.add_argument("--no-sandbox")
            self.options.add_argument("--disable-dev-shm-usage")
            
            self.driver = uc.Chrome(executable_path=chromedriver_path, options=self.options, use_subprocess=True, enable_cdp_events=True)
            print(f"chromedriver path: {chromedriver_path}")
            self.wait = WebDriverWait(self.driver, 10)
    
    def login(self, username, password):
        self.driver.get('https://www.linkedin.com/login')
        email_field = self.wait.until(EC.presence_of_element_located((By.ID, 'username')))
        password_field = self.driver.find_element(By.ID, 'password')

        email_field.send_keys(username)
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)

        # Wait until login is successful or verification is required
        feed_url = "https://www.linkedin.com/feed/"
        max_wait_time = 120  # Maximum wait time in seconds
        wait_interval = 5  # Time to wait between checks in seconds

        start_time = time.time()
        while True:
            current_url = self.driver.current_url
            if current_url == feed_url:
                print("Login successful.")
                break
            elif time.time() - start_time > max_wait_time:
                raise ValueError("Login timeout: Verification not completed in time.")
            else:
                print("Waiting for user to complete verification...")
                time.sleep(wait_interval)

    def search_for_company(self, company, team):
        search_query = f"{company} {team}"
        search_input = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.search-global-typeahead__input")))
        search_input.clear()
        search_input.send_keys(search_query)
        search_input.send_keys(Keys.RETURN)
    
    def click_employees_link(self):
        employees_link = self.wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/search/results/people/?currentCompany=')]")))
        employees_link.click()
        
    def click_peoples_button(self):
        try:
            peoples_button = self.wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'search-results__cluster-bottom-banner')]//a[contains(text(), 'See all people results')]")))
            peoples_button.click()
        except Exception as e:
            print(f"Failed to click on 'See all people results' button: {e}")
    
    def open_all_filters(self):
        all_filters_button = self.wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(@aria-label, 'Show all filters')]")))
        all_filters_button.click()
    
    def populate_filters(self, location=None, current_company=None, team=None):
        print("populate filters called!")

        if location:
            try:
                # Click on "Add a location" button
                add_location_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@data-add-filter-button, '') and .//span[text()='Add a location']]"))
                )
                self.driver.execute_script("arguments[0].click();", add_location_button)

                # Type the location
                location_input = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@aria-label='Add a location']")))
                location_chunks = location.split(", ")
                base_location = ""
                if location_chunks:
                    base_location += location_chunks[0]
                    for chunk in location_chunks[1:]:
                        if len(chunk) > 2: # To prevent things like New York, NY but still concatenate sub-geosections
                            base_location += f", {chunk}"
                    print(f"base location for {current_company}: {base_location}")
                    location_input.send_keys(base_location)
                else:
                    location_input.send_keys(location)
                time.sleep(1)  # Allow time for the search results to populate
                print("Location input populated")

                # Select the top first result if it exists
                search_results = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'basic-typeahead__selectable')]")
                if search_results:
                    print(f"location search results: {search_results}")
                    self.driver.execute_script("arguments[0].click();", search_results[0])
            except Exception as e:
                print(f"An error occurred while setting location filter: {e}")

    def apply_filters(self):
        print("Apply Filters called!")
        apply_button = self.wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(@aria-label, 'Apply current filters to show results')]")))
        self.driver.execute_script("arguments[0].click();", apply_button)
    
    def get_profiles(self, company: str, job_id: str):
        time.sleep(2)
        
        profile_links = []
        profile_data = []
        page_count = 0  # Initialize a page counter
        company_name = company.lower()
        company_set = {company_name, f"at {company_name}", f"at{company_name}", f"@ {company_name}", f"@{company_name}"}
        image_dir = f"./database/{job_id}/"
        
        # Ensure the directory exists
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        while page_count < 4:  # Limit to the first 4 pages
            try:
                # Wait until profile links are present
                profiles = self.wait.until(
                                EC.presence_of_all_elements_located(
                                    (By.XPATH, "//a[contains(@href, '/in/') and not(parent::*[starts-with(@class, 'reusable')])]")
                                )
                            )
            except Exception as e:
                print(f"An error occurred while fetching profiles at current page: {e}")
                break
            # self.wait.until(EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@href, '/in/')]")))
            
            # Collect profile URLs from the current page
            for profile in profiles:
                try:
                    # profile url
                    profile_url = profile.get_attribute('href')
                    
                    # name element for ith profile
                    name = ""
                    try:
                        name_element = profile.find_element(By.XPATH, ".//span[contains(@aria-hidden, 'true')]")
                        name = name_element.text
                        print(f"fetched name in get_profiles: {name}")
                    except Exception:
                        pass
                    
                    # Find the nearest 'li' parent element
                    li_element = profile.find_element(By.XPATH, "./ancestor::li[contains(@class, 'reusable-search__result-container')]")
                    print(f"profile container element class name: {li_element.get_attribute('class')}")
                    
                    # PROFILE IMAGE
                    image_path = None
                    try:
                        # Profile image container
                        profile_image_container = li_element.find_element(By.XPATH, ".//div[contains(@class, 'presence-entity')]")
                        # Image element
                        image_element = profile_image_container.find_element(By.TAG_NAME, "img")
                        image_url = image_element.get_attribute('src')
                        
                        # Download the image
                        try:
                            image_response = requests.get(image_url, timeout=10)
                            # Check if the response was successful (status code 200)
                            image_response.raise_for_status()
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
                        # profile_image = Image.open(BytesIO(image_response.content))
                        image_name = f"{profile_url.split('/')[-1].split('?')[0]}.png"
                        image_path = os.path.join(image_dir, image_name)

                        with open(image_path, 'wb') as f:
                            f.write(image_response.content)
                    
                    except Exception as e:
                        print(f"failed to extract profile image: {e}")
                    
                    # Summary elements for ith profile
                    summary_elements = li_element.find_elements(By.XPATH, ".//p[contains(@class, 'entity-result__summary')]")
                    # Subtitle elements for ith profile
                    subtitle_elements = li_element.find_elements(By.XPATH, ".//div[contains(@class, 'entity-result__primary-subtitle')]")
                    
                    company_in_profile = False
                    for subtitle_element in subtitle_elements:
                        subtitle_text = subtitle_element.text.lower()
                        
                        strong_elements = subtitle_element.find_elements(By.TAG_NAME, "strong")
                        for search_string in company_set:
                            if any(search_string in strong.text.lower() for strong in strong_elements):
                                company_in_profile = True
                                break
                            if search_string in subtitle_text:
                                company_in_profile = True
                                break
                        if company_in_profile:
                            break
                        
                    if company_in_profile:
                        print("Adding profile url!")
                        profile_links.append(profile_url)
                        profile_data.append({"url" : profile_url, "image_path": image_path, "name": name})
                        global_profile_urls[job_id] = profile_links
                    else:
                        for summary_element in summary_elements:
                            summary_text = summary_element.text.lower()
                            
                            strong_elements = summary_element.find_elements(By.TAG_NAME, "strong")
                            for search_string in company_set:
                                if any(search_string in strong.text.lower() for strong in strong_elements):
                                    company_in_profile = True
                                    if "current" not in summary_text:
                                        company_in_profile = False
                                    break
                                if search_string in summary_text:
                                    company_in_profile = True
                                    if "current" not in summary_text:
                                        company_in_profile = False
                                    break
                            if company_in_profile:
                                break

                        if company_in_profile:
                            print("Adding profile url!")
                            profile_links.append(profile_url)
                            profile_data.append({"url" : profile_url, "image_path": image_path, "name": name})
                            global_profile_urls[job_id] = profile_links

                except Exception as e:
                    print(f"An error occurred while verifying the profile: {e}")

            try:
                # Scroll to the bottom of the page to ensure the "Next" button appears
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Locate the "Next" button and check if it's enabled
                next_button = self.wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(@class, 'artdeco-pagination__button--next')]")))
                if "artdeco-button--disabled" in next_button.get_attribute("class"):
                    print("Next button is disabled. End of pagination.")
                    break  # Exit the loop if the "Next" button is disabled
                
                # Click the "Next" button
                self.driver.execute_script("arguments[0].click();", next_button)
                
                # Wait for the next page to load
                time.sleep(2)  # Adjust sleep time if necessary to wait for the next page to load
                
                page_count += 1  # Increment the page counter
                
            except Exception as e:
                print(f"An error occurred while navigating to the next page: {e}")
                break  # Exit the loop if there is an error locating or clicking the "Next" button

        return profile_data

    def close(self):
        self.driver.quit()
        
    def minimize(self):
        self.driver.minimize_window()

class Extractor:
    scraper: LinkedInProfileScraper = None
    
    extract_relevant_search_tags_json = [
        {
            "type": "function",
            "function": {
                "name": "extract_relevant_search_tags",
                "description": ("Given a vector embedded website for a job posting page, extract these Linkedin search tags: "
                                "1. Location"
                                "2. Current Company"
                                "3. Title"
                                "4. Team"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Physical location of the job described by the job posting"
                        },
                        "current_company": {
                            "type": "string",
                            "description": "Company of the job posting"
                        },
                        "title": {
                            "type": "string",
                            "description": "Title of the job in the job posting page"
                        },
                        "team": {
                            "type": "string",
                            "description": "Single-word keyword for the Team the job applicant of the job posting would be part of - For instance, executive assistant would be part of the executive team, software engineer would be part of the software team, and Machine Learning Engineer would be part of the AI team"
                        }
                    },
                    "required": ["location", "current_company", "title", "team"]
                }
            }
        }
    ]
    
    extract_relevant_profile_keywords_json = [
        {
            "type": "function",
            "function": {
                "name": "extract_relevant_profile_keywords",
                "description": ("Given a vector embedded LinkedIn profile page, extract these profile keywords: "
                                "1. Name"
                                "2. Current Company"
                                "3. Title"
                                "4. Team"
                                "5. Most Recent School"
                                "6. Undergraduate School"
                                "7. Total Years Employed"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the profile"
                        },
                        "current_company": {
                            "type": "string",
                            "description": "Current company profile works at"
                        },
                        "title": {
                            "type": "string",
                            "description": "Title of profile"
                        },
                        "team": {
                            "type": "string",
                            "description": "Team profile is part of"
                        },
                        "most_recent_school": {
                            "type": "string",
                            "description": "Most recent school attended by the profile"
                        },
                        "undergraduate_school": {
                            "type": "string",
                            "description": "Undergraduate school attended by the profile"
                        },
                        "total_years_employed": {
                            "type": "string",
                            "description": "Total years of employment of the profile"
                        }
                    },
                    "required": ["name", "current_company", "title", "team", "most_recent_school", "undergraduate_school", "total_years_employed"]
                }
            }
        }
    ]
    
    def extract_relevant_search_tags(self, id: str, location: str, current_company: str, title: str, team: str) -> List[str]:
        """Method called by an LLM once it extracts relevant Linkedin search tags from its input"""
        # Replace this with the LinkedIn company URL
        company_url = f"https://www.linkedin.com/company/{'-'.join(current_company.split(' ')).lower()}/" # Find governing logic
        # print(f"company url: {company_url}")
        
        # self.scraper.login(username, password)
        self.scraper.driver.get('https://www.linkedin.com/feed/')
        self.scraper.search_for_company(current_company, team)
        self.scraper.click_peoples_button()
        self.scraper.open_all_filters()
        self.scraper.populate_filters(location=location, current_company=current_company, team=team)
        self.scraper.apply_filters()
        
        # profiles is a list with a dict of url and image as each element
        profiles = self.scraper.get_profiles(current_company, id)
        # print(f"extracted profiles: {profiles}")
        
        # Save job_profiles_dict as json for future fetching with unique_id
        job_data_file = f"./database/job_profiles_{id}.json"
        job_profiles_dict = {id: profiles}
        with open(job_data_file, "w", encoding="utf-8") as f:
            json.dump(job_profiles_dict, f)
        
        self.scraper.minimize()
        
        print(f"extracted keywords: {[location, current_company, title, team]}")
        
        return [location, current_company, title]
    
    def extract_relevant_profile_keywords(self, url: str, id: str, name: str, current_company: str, title: str, team: str, most_recent_school: str, undergraduate_school: str, total_years_employed: str, image_path=None) -> dict:
        """Method called by an LLM once it extracts relevant profile keywords from its input"""
        # Create a profile object
        profile_object = {
            "url": url,
            "image_path": image_path,
            "id": id,
            "name": name,
            "current_company": current_company,
            "title": title,
            "team": team,
            "most_recent_school": most_recent_school,
            "undergraduate_school": undergraduate_school,
            "total_years_employed": total_years_employed
        }
            
        profile_pdf_basename = get_pdf_path(id)
        print(f"profile pdf basename for {url}: {profile_pdf_basename} with id: {id}")
        
        return {profile_pdf_basename: profile_object}
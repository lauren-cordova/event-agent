### IMPORTS & SETUP ###

# Standard library imports
import os
import sys
import re
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin
import json

# Third-party imports
import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle
import base64

# Import Secrets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../secrets')))
import my_secrets

### FUNCTIONS ###

# Check if all required packages are installed and up to date
def check_requirements():
    import subprocess
    import pkg_resources
    
    # Required packages from requirements.txt
    required_packages = {
        'streamlit': '1.46.1',
        'google-api-python-client': '2.174.0',
        'google-auth-httplib2': '0.2.0',
        'google-auth-oauthlib': '1.2.2',
        'openai': '1.92.2',
        'beautifulsoup4': '4.13.4',
        'requests': '2.32.4',
        'lxml': '4.9.3'
    }
    
    outdated_packages = []
    missing_packages = []
    
    for package, required_version in required_packages.items():
        try:
            # Get installed version
            installed_version = pkg_resources.get_distribution(package).version
            installed = pkg_resources.parse_version(installed_version)
            required = pkg_resources.parse_version(required_version)
            
            if installed < required:
                outdated_packages.append(f"{package} (installed: {installed_version}, required: {required_version})")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
    
    return missing_packages, outdated_packages

# Connect to Gmail and get emails
def process_emails():
    try:
        # Gmail API setup
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/spreadsheets']
        creds = None
        
        # Load existing credentials
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        # Build Gmail service
        service = build('gmail', 'v1', credentials=creds)
        
        # Get emails from inbox
        results = service.users().messages().list(userId='me', labelIds=['INBOX']).execute()
        messages = results.get('messages', [])
        
        if not messages:
            st.success("No new emails found in inbox.")
            return
        
        # Google Sheets setup
        sheets_service = build('sheets', 'v4', credentials=creds)
        spreadsheet_id = my_secrets.SPREADSHEET_ID
        
        # Get existing URLs from sheet
        try:
            result = sheets_service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range='Events!K:K'
            ).execute()
            existing_urls = [row[0] for row in result.get('values', [])[1:] if row]  # Skip header
            next_row = len(existing_urls) + 2  # +2 because sheet is 1-indexed and we skip header
        except:
            existing_urls = []
            next_row = 2  # Start at row 2 (after header)
        
        new_urls_count = 0
        
        for message in messages:  # Process all emails
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            # Extract email content
            payload = msg['payload']
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/html':
                        data = part['body']['data']
                        html_content = base64.urlsafe_b64decode(data).decode('utf-8')
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Extract URLs from HTML
                        urls = []
                        for link in soup.find_all('a', href=True):
                            url = link['href']
                            if url.startswith('http'):
                                # Remove URL parameters
                                clean_url = url.split('?')[0]
                                if clean_url not in existing_urls and clean_url not in urls:
                                    urls.append(clean_url)
                        
                        # Also extract URLs from text content
                        text_content = soup.get_text()
                        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                        text_urls = re.findall(url_pattern, text_content)
                        for url in text_urls:
                            clean_url = url.split('?')[0]
                            if clean_url not in existing_urls and clean_url not in urls:
                                urls.append(clean_url)
                        
                        # Add new URLs to sheet
                        for url in urls:
                            try:
                                # Update the specific cell in column K using the calculated next_row
                                sheets_service.spreadsheets().values().update(
                                    spreadsheetId=spreadsheet_id,
                                    range=f'Events!K{next_row}',
                                    valueInputOption='RAW',
                                    body={'values': [[url]]}
                                ).execute()
                                existing_urls.append(url)
                                next_row += 1  # Increment for next URL
                                new_urls_count += 1
                            except Exception as e:
                                st.error(f"Error adding URL to sheet: {e}")
                        break
                
                # Mark email as read and archive it
                try:
                    # Remove UNREAD and INBOX labels (removing INBOX archives the email)
                    service.users().messages().modify(
                        userId='me',
                        id=message['id'],
                        body={
                            'removeLabelIds': ['UNREAD', 'INBOX']
                        }
                    ).execute()
                except Exception as e:
                    st.warning(f"Could not archive email {message['id']}: {e}")
        
        st.success(f"Processed emails and added {new_urls_count} new unique URLs to the sheet.")
        
    except Exception as e:
        st.error(f"Error processing emails: {e}")

# Scrape event pages and get event data
def populate_events():
    try:
        # Setup OpenAI
        openai.api_key = my_secrets.openai_by
        
        # Google Sheets setup
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/spreadsheets']
        creds = None
        
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        sheets_service = build('sheets', 'v4', credentials=creds)
        spreadsheet_id = my_secrets.SPREADSHEET_ID
        
        # Get URLs from sheet
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range='Events!K:K'
        ).execute()
        
        urls = [row[0] for row in result.get('values', [])[1:] if row]  # Skip header
        
        if not urls:
            st.warning("No URLs found in the sheet to process.")
            return
        
        processed_count = 0
        
        # Process URLs with a small delay to avoid rate limits
        for i, url in enumerate(urls):
            try:
                # Add a small delay between processing to avoid rate limits
                if i > 0:
                    time.sleep(2)  # 2 second delay between URLs
                
                # Scrape webpage
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text_content = soup.get_text()
                text_content = ' '.join(text_content.split())[:9999]  # Limit to 9999 chars
                
                # Use ChatGPT to extract event data
                prompt = f"""
                Analyze this webpage content and extract event information. You must respond with ONLY a valid JSON object, no other text.

                Return this exact JSON structure:
                {{
                    "event": true/false,
                    "event_name": "Event name or title",
                    "date": "Event date, report results in mm/dd/yyyy format",
                    "start_time": "Event start time, report results in hh:mm AM/PM format", 
                    "end_time": "Event end time, report results in hh:mm AM/PM format",
                    "city": "City name",
                    "state": "2 letter state abbreviation",
                    "venue": "Venue name",
                    "address": "Full address",
                    "description": "Brief description of the event"
                }}

                Rules:
                - Set "event" to true only if this is clearly an event page
                - For date extraction: Look for actual event dates mentioned in the content (like "July 9") and assume year is 2025 if not specified
                - Prioritize event date and start/end times from the top of the page and not in the event description section which may reference other things
                - Do NOT guess or assume dates, or default to July 9, only use dates explicitly mentioned in the content
                - If any field cannot be found, use empty string ""
                - Return ONLY the JSON object, no explanations or extra text
                - Ensure the JSON is properly formatted and date is in mm/dd/yyyy format

                Webpage content: {text_content}
                """
                
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                
                # Parse JSON response
                try:
                    # Clean the response to extract just the JSON
                    response_text = response.choices[0].message.content.strip()
                    
                    # Debug: Show what we received
                    st.info(f"Raw response for {url}: {response_text[:200]}...")
                    
                    # Try to find JSON object in the response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        event_data = json.loads(json_text)
                        st.success(f"Successfully parsed JSON for {url}")
                    else:
                        # If no JSON found, create default structure
                        st.warning(f"Could not find JSON in response for URL: {url}")
                        st.warning(f"Full response: {response_text}")
                        event_data = {
                            'event': False,
                            'event_name': '',
                            'date': '',
                            'start_time': '',
                            'end_time': '',
                            'city': '',
                            'state': '',
                            'venue': '',
                            'address': '',
                            'description': ''
                        }
                    
                    # Update the corresponding row in the sheet
                    row_number = i + 2  # +2 because sheet is 1-indexed and we skip header
                    
                    # Create a single row with all event data
                    event_row = [
                        event_data.get('event', ''),
                        event_data.get('event_name', ''),
                        event_data.get('date', ''),
                        event_data.get('start_time', ''),
                        event_data.get('end_time', ''),
                        event_data.get('city', ''),
                        event_data.get('state', ''),
                        event_data.get('venue', ''),
                        event_data.get('address', ''),
                        event_data.get('description', '')
                    ]
                    
                    # Update all columns in a single API call
                    sheets_service.spreadsheets().values().update(
                        spreadsheetId=spreadsheet_id,
                        range=f'Events!A{row_number}:J{row_number}',
                        valueInputOption='RAW',
                        body={'values': [event_row]}
                    ).execute()
                    
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse JSON response for URL: {url}. Error: {e}")
                    st.error(f"Response was: {response.choices[0].message.content.strip()}")
                    continue
                    
            except Exception as e:
                st.error(f"Error processing URL {url}: {e}")
                continue
        
        st.success(f"Successfully processed {processed_count} URLs and populated event data.")
        
    except Exception as e:
        st.error(f"Error populating events: {e}")

# Check and validate event data accuracy
def check_output():
    try:
        # Setup OpenAI
        openai.api_key = my_secrets.openai_by
        
        # Google Sheets setup
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/spreadsheets']
        creds = None
        
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        sheets_service = build('sheets', 'v4', credentials=creds)
        spreadsheet_id = my_secrets.SPREADSHEET_ID
        
        # Get all data from sheet
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range='Events!A:K'
        ).execute()
        
        rows = result.get('values', [])
        if len(rows) < 2:  # Need at least header + 1 data row
            st.warning("No data found in sheet to validate.")
            return
        
        # Skip header row
        data_rows = rows[1:]
        corrected_count = 0
        checked_count = 0
        
        for i, row in enumerate(data_rows):
            # Ensure row has enough columns
            while len(row) < 11:
                row.append('')
            
            event_bool = row[0] if len(row) > 0 else ''
            url = row[10] if len(row) > 10 else ''  # Column K
            
            # Only check rows where event=true and URL exists
            if event_bool == 'TRUE' and url and url.startswith('http'):
                checked_count += 1
                try:
                    # Scrape webpage
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text_content = soup.get_text()
                    text_content = ' '.join(text_content.split())[:9999]  # Limit to 9999 chars
                    
                    # Use ChatGPT to extract date and time specifically
                    prompt = f"""
                    Analyze this webpage content and extract ONLY the event date and times. You must respond with ONLY a valid JSON object, no other text.

                    Return this exact JSON structure:
                    {{
                        "date": "Event date, report results in mm/dd/yyyy format",
                        "start_time": "Event start time, report results in hh:mm AM/PM format", 
                        "end_time": "Event end time, report results in hh:mm AM/PM format"
                    }}

                    Rules:
                    - Look for actual event dates mentioned in the content (like "July 9", "7/9", "July 9th", etc.)
                    - Assume year is 2025 if not specified
                    - Look for actual event times mentioned in the content
                    - Prioritize date and time information from the top of the page
                    - Make sure the date and time reflects when the event is happening, and NOT some other reference that may be described in the content
                    - If no specific date/time is mentioned, use empty string ""
                    - Return ONLY the JSON object, no explanations or extra text
                    - Ensure the JSON is properly formatted and date is in mm/dd/yyyy format

                    Webpage content: {text_content}
                    """
                    
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.1
                    )
                    
                    # Parse JSON response
                    try:
                        response_text = response.choices[0].message.content.strip()
                        
                        # Try to find JSON object in the response
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        
                        if json_start != -1 and json_end > json_start:
                            json_text = response_text[json_start:json_end]
                            validation_data = json.loads(json_text)
                            
                            # Get current values from sheet
                            current_date = row[2] if len(row) > 2 else ''  # Column C
                            current_start_time = row[3] if len(row) > 3 else ''  # Column D
                            current_end_time = row[4] if len(row) > 4 else ''  # Column E
                            
                            # Check if validation data is different from current data
                            new_date = validation_data.get('date', '')
                            new_start_time = validation_data.get('start_time', '')
                            new_end_time = validation_data.get('end_time', '')
                            
                            if (new_date and new_date != current_date) or \
                               (new_start_time and new_start_time != current_start_time) or \
                               (new_end_time and new_end_time != current_end_time):
                                
                                # Update the row with corrected data
                                row_number = i + 2  # +2 because sheet is 1-indexed and we skip header
                                
                                # Create updated row data
                                updated_row = [
                                    row[0],  # event (keep as is)
                                    row[1],  # event_name (keep as is)
                                    new_date if new_date else current_date,  # date
                                    new_start_time if new_start_time else current_start_time,  # start_time
                                    new_end_time if new_end_time else current_end_time,  # end_time
                                    row[5] if len(row) > 5 else '',  # city (keep as is)
                                    row[6] if len(row) > 6 else '',  # state (keep as is)
                                    row[7] if len(row) > 7 else '',  # venue (keep as is)
                                    row[8] if len(row) > 8 else '',  # address (keep as is)
                                    row[9] if len(row) > 9 else '',  # description (keep as is)
                                    row[10] if len(row) > 10 else ''  # URL (keep as is)
                                ]
                                
                                # Update the row
                                sheets_service.spreadsheets().values().update(
                                    spreadsheetId=spreadsheet_id,
                                    range=f'Events!A{row_number}:K{row_number}',
                                    valueInputOption='RAW',
                                    body={'values': [updated_row]}
                                ).execute()
                                
                                corrected_count += 1
                                st.success(f"Corrected data for row {row_number}: {url}")
                                
                        else:
                            st.warning(f"Could not parse validation response for URL: {url}")
                            
                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse validation JSON for URL: {url}. Error: {e}")
                        continue
                        
                except Exception as e:
                    st.error(f"Error validating URL {url}: {e}")
                    continue
        
        st.success(f"Validation complete. Checked {checked_count} event rows and corrected {corrected_count} rows with updated date/time information.")
        
    except Exception as e:
        st.error(f"Error checking output: {e}")

### MAIN ###

# Streamlit app
def main():
    st.set_page_config(
        page_title="Event Agent",
        page_icon="🎫",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .button-container {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 2rem 0;
    }
    .link-container {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎫 Event Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Automatically extract event URLs from emails and populate event data from web pages</p>', unsafe_allow_html=True)
    
    # Main buttons stacked vertically
    if st.button("📧 Process Emails", use_container_width=True):
        with st.spinner("Processing emails..."):
            process_emails()
    
    if st.button("🌐 Scrape Event Data", use_container_width=True):
        with st.spinner("Scraping event data..."):
            populate_events()
    
    if st.button("✅ Validate & Correct Data", use_container_width=True):
        with st.spinner("Validating and correcting event data..."):
            check_output()
    
    # Left sidebar for Quick Links and System Status
    with st.sidebar:
        st.markdown("## 🔗 Quick Links")
        
        # Gmail button
        if st.button("📧 Open Gmail", use_container_width=True):
            st.markdown(f'<a href="https://mail.google.com" target="_blank">Opening Gmail...</a>', unsafe_allow_html=True)
        
        # Google Sheets button
        if st.button("📊 Open Google Sheet", use_container_width=True):
            st.markdown(f'<a href="https://docs.google.com/spreadsheets/d/{my_secrets.SPREADSHEET_ID}" target="_blank">Opening Google Sheet...</a>', unsafe_allow_html=True)
        
        # Refresh Gmail Connection button
        if st.button("🔄 Refresh Gmail Connection", use_container_width=True):
            if os.path.exists('token.pickle'):
                os.remove('token.pickle')
                st.success("Gmail connection refreshed. Please try processing emails again.")
            else:
                st.info("No existing connection to refresh.")
        
        st.markdown("---")
        st.markdown("## 📊 System Status")
        
        # Check requirements
        missing_packages, outdated_packages = check_requirements()
        
        if missing_packages:
            st.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        else:
            st.success("✅ All required packages are installed")
        
        if outdated_packages:
            st.warning(f"⚠️ Outdated packages: {', '.join(outdated_packages)}")
        else:
            st.success("✅ All packages are up to date")
        
        # Check if credentials file exists
        if os.path.exists('credentials.json'):
            st.success("✅ Gmail API credentials found")
        else:
            st.error("❌ Gmail API credentials not found. Please add credentials.json file.")
        
        # Check if secrets are configured
        try:
            if hasattr(my_secrets, 'SPREADSHEET_ID') and my_secrets.SPREADSHEET_ID != "your_spreadsheet_id_here":
                st.success("✅ Google Sheets ID configured")
            else:
                st.error("❌ Google Sheets ID not configured in my_secrets.py")
        except:
            st.error("❌ my_secrets.py not found or not properly configured")
        
        try:
            if hasattr(my_secrets, 'openai_by') and my_secrets.openai_by != "your_openai_api_key":
                st.success("✅ OpenAI API key configured")
            else:
                st.error("❌ OpenAI API key not configured in my_secrets.py")
        except:
            st.error("❌ OpenAI API key not configured")

if __name__ == "__main__":
    main()
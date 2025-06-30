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
                range='Events!J:J'
            ).execute()
            existing_urls = [row[0] for row in result.get('values', [])[1:] if row]  # Skip header
        except:
            existing_urls = []
        
        new_urls_count = 0
        
        for message in messages[:10]:  # Process last 10 emails
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
                                # Find the next empty row in column J
                                result = sheets_service.spreadsheets().values().get(
                                    spreadsheetId=spreadsheet_id,
                                    range='Events!J:J'
                                ).execute()
                                values = result.get('values', [])
                                next_row = len(values) + 1  # +1 because sheets are 1-indexed
                                
                                # Update the specific cell in column J
                                sheets_service.spreadsheets().values().update(
                                    spreadsheetId=spreadsheet_id,
                                    range=f'Events!J{next_row}',
                                    valueInputOption='RAW',
                                    body={'values': [[url]]}
                                ).execute()
                                existing_urls.append(url)
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
            range='Events!J:J'
        ).execute()
        
        urls = [row[0] for row in result.get('values', [])[1:] if row]  # Skip header
        
        if not urls:
            st.warning("No URLs found in the sheet to process.")
            return
        
        processed_count = 0
        
        for i, url in enumerate(urls):
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
                text_content = ' '.join(text_content.split())[:3000]  # Limit to 3000 chars
                
                # Use ChatGPT to extract event data
                prompt = f"""
                Extract event information from this webpage content. Return ONLY a JSON object with these exact fields:
                {{
                    "event_name": "Event name or title",
                    "date": "Date that the event takes place, assume year is 2025 if not specified, report results in mm/dd/yyyy format",
                    "start_time": "Start time of the event, report results in hh:mm AM/PM format", 
                    "end_time": "End time of the event, report results in hh:mm AM/PM format",
                    "city": "City name",
                    "state": "2 letter state abbreviation",
                    "venue": "Venue name",
                    "address": "Full address",
                    "description": "Brief description of the event"
                }}
                
                If any field cannot be found, leave it blank.
                
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
                    event_data = json.loads(response.choices[0].message.content.strip())
                    
                    # Update the corresponding row in the sheet
                    row_number = i + 2  # +2 because sheet is 1-indexed and we skip header
                    
                    # Create a single row with all event data
                    event_row = [
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
                        range=f'Events!A{row_number}:I{row_number}',
                        valueInputOption='RAW',
                        body={'values': [event_row]}
                    ).execute()
                    
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    st.error(f"Failed to parse JSON response for URL: {url}")
                    continue
                    
            except Exception as e:
                st.error(f"Error processing URL {url}: {e}")
                continue
        
        st.success(f"Successfully processed {processed_count} URLs and populated event data.")
        
    except Exception as e:
        st.error(f"Error populating events: {e}")

### MAIN ###

# Streamlit app
def main():
    st.set_page_config(
        page_title="Event Scraper",
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
    
    st.markdown('<h1 class="main-header">🎫 Event Scraper</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Automatically extract event URLs from emails and populate event data from web pages</p>', unsafe_allow_html=True)
    
    # Main buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Process Emails", use_container_width=True):
            with st.spinner("Processing emails..."):
                process_emails()
    
    with col2:
        if st.button("🌐 Scrape Event Data", use_container_width=True):
            with st.spinner("Scraping event data..."):
                populate_events()
    
    with col3:
        if st.button("🔄 Refresh Gmail Connection", use_container_width=True):
            if os.path.exists('token.pickle'):
                os.remove('token.pickle')
                st.success("Gmail connection refreshed. Please try processing emails again.")
            else:
                st.info("No existing connection to refresh.")
    
    # Links section
    st.markdown("---")
    st.markdown('<h3 style="text-align: center;">Quick Links</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="https://mail.google.com" target="_blank" style="text-decoration: none;">
                <button style="background-color: #ea4335; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: bold; cursor: pointer;">
                    📧 Open Gmail
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="https://docs.google.com/spreadsheets/d/{my_secrets.SPREADSHEET_ID}" target="_blank" style="text-decoration: none;">
                <button style="background-color: #0f9d58; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: bold; cursor: pointer;">
                    📊 Open Google Sheet
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Status section
    st.markdown("---")
    st.markdown('<h3 style="text-align: center;">System Status</h3>', unsafe_allow_html=True)
    
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
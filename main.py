# Standard library imports
import sys
import os
import time
import json
import re
import base64
import uuid
from datetime import datetime

# Third-party imports
import streamlit as st
import boto3
import pandas as pd
import subprocess
import requests
from bs4 import BeautifulSoup
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import email.utils

# Import Secrets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../secrets')))
import my_secrets

# Constants
GOOGLE_SHEETS_CHAR_LIMIT = 50000
GOOGLE_SHEETS_TRUNCATE_LIMIT = 49985
TRUNCATE_SUFFIX = "... (truncated)"

# Gmail API SCOPES
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/spreadsheets'
]

# Initialize clients
dynamodb = boto3.resource('dynamodb',
    aws_access_key_id=my_secrets.event_agent_aws_access_key_id,
    aws_secret_access_key=my_secrets.event_agent_aws_secret_access_key,
    region_name=my_secrets.event_agent_aws_region
)

events_table = dynamodb.Table('events')
qdrant_client = QdrantClient(url=my_secrets.QDRANT_URL, api_key=my_secrets.QDRANT_API_KEY)
openai_client = OpenAI(api_key=my_secrets.openai_by)

# ============================================================================
# AUTHENTICATION & SERVICE FUNCTIONS
# ============================================================================

def load_credentials():
    """Load or refresh Google API credentials."""
    creds = None
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if not all(scope in creds.scopes for scope in SCOPES):
                print("Missing required scopes, refreshing credentials...")
                creds = None
        except Exception as e:
            print(f"Error loading credentials: {e}")
            creds = None
    
    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                print("No valid credentials found, starting OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            print("Credentials saved successfully")
        except Exception as e:
            print(f"Error in OAuth flow: {e}")
            raise Exception("Failed to obtain valid credentials. Please try again.")
    
    return creds

def get_service(service_name):
    """Get Google API service instance for specified service."""
    try:
        creds = load_credentials()
        if service_name == "gmail":
            return build('gmail', 'v1', credentials=creds)
        elif service_name == "sheets":
            return build('sheets', 'v4', credentials=creds)
        else:
            print(f"Unknown service: {service_name}")
            return None
    except Exception as e:
        print(f"Error getting {service_name} service: {str(e)}")
        return None

def check_config(tool):
    """Check configuration status for various tools."""
    try:
        if tool == "gmail":
            service = get_service("gmail")
            if service:
                service.users().labels().list(userId='me').execute()
                return True
            return False
        elif tool == "qdrant":
            if not hasattr(my_secrets, 'QDRANT_URL') or not hasattr(my_secrets, 'QDRANT_API_KEY'):
                return False
            client = QdrantClient(url=my_secrets.QDRANT_URL, api_key=my_secrets.QDRANT_API_KEY)
            client.get_collections()
            # Also ensure the emails collection exists
            return ensure_qdrant_collection()
        elif tool == "aws":
            dynamodb.meta.client.list_tables()
            return True
        elif tool == "dynamo":
            existing_tables = dynamodb.meta.client.list_tables()['TableNames']
            return 'events' in existing_tables
        elif tool == "google_sheets":
            service = get_service("sheets")
            if service:
                service.spreadsheets().get(spreadsheetId=my_secrets.SPREADSHEET_ID).execute()
                return True
            return False
        else:
            print(f"Unknown tool: {tool}")
            return False
    except Exception as e:
        print(f"Error checking {tool} config: {str(e)}")
        return False 

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def process_datetime(datetime_str, output_format="timestamp"):
    """Unified datetime processing function with multiple output formats."""
    try:
        if output_format == "timestamp":
            formats_to_try = [
                '%m/%d/%Y %I:%M %p',  # Full datetime with AM/PM
                '%m/%d/%Y %H:%M',     # Full datetime with 24-hour format
                '%m/%d/%Y',           # Date only
                '%Y-%m-%d %H:%M:%S',  # ISO format
                '%Y-%m-%d',           # ISO date only
            ]
            
            for fmt in formats_to_try:
                try:
                    dt = datetime.strptime(datetime_str, fmt)
                    timestamp = int(dt.timestamp())
                    return timestamp
                except ValueError:
                    continue
            
            # Try email date format (RFC 2822)
            try:
                parsed_date = email.utils.parsedate_tz(datetime_str)
                if parsed_date:
                    timestamp = int(email.utils.mktime_tz(parsed_date))
                    return timestamp
            except Exception as e:
                print(f"Failed to parse email date '{datetime_str}': {e}")
            
            # Return current timestamp if parsing fails
            current_timestamp = int(time.time())
            print(f"Could not parse datetime string: {datetime_str}, using current timestamp: {current_timestamp}")
            return current_timestamp
            
        elif output_format == "formatted":
            # Parse email date string and format as MM/DD/YYYY HH:MM AM/PM
            parsed_date = email.utils.parsedate_tz(datetime_str)
            if parsed_date:
                dt = datetime.fromtimestamp(email.utils.mktime_tz(parsed_date))
                return dt.strftime('%m/%d/%Y %I:%M %p')
            return datetime_str
        elif output_format == "date_only":
            # Try to extract just the date part from various formats
            formats_to_try = [
                '%m/%d/%Y %I:%M %p',  # Full datetime with AM/PM
                '%m/%d/%Y %H:%M',     # Full datetime with 24-hour format
                '%m/%d/%Y',           # Date only
                '%Y-%m-%d %H:%M:%S',  # ISO format
                '%Y-%m-%d',           # ISO date only
            ]
            
            for fmt in formats_to_try:
                try:
                    dt = datetime.strptime(datetime_str, fmt)
                    return dt.strftime('%m/%d/%Y')
                except ValueError:
                    continue
            
            print(f"Could not parse datetime string for date extraction: {datetime_str}")
            return datetime_str
        else:
            print(f"Unknown output format: {output_format}")
            return datetime_str
    except Exception as e:
        print(f"Error processing datetime: {str(e)}")
        if output_format == "timestamp":
            return int(time.time())
        return datetime_str

def sanitize_url(url):
    """Sanitize URL to ensure proper format."""
    if not url:
        return ""
    # Remove any query parameters
    url = url.split('?')[0]
    return url

def run_setup_script():
    """Run the setup script."""
    try:
        result = subprocess.run(['python3', 'setup.py'], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Setup completed successfully!")
            print("Setup output:", result.stdout)
        else:
            st.error(f"Error during setup: {result.stderr}")
            print("Setup error output:", result.stderr)
    except Exception as e:
        st.error(f"Error running setup script: {e}") 

# ============================================================================
# QDRANT VECTOR DATABASE FUNCTIONS
# ============================================================================

def ensure_qdrant_collection():
    """Ensure the Qdrant emails collection exists."""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_exists = any(col.name == "emails" for col in collections)
        
        if not collection_exists:
            print("Creating 'emails' collection in Qdrant...")
            qdrant_client.create_collection(
                collection_name="emails",
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embedding size
                    distance=models.Distance.COSINE
                )
            )
            print("✅ Collection 'emails' created successfully!")
        else:
            print("✅ Collection 'emails' already exists.")
        
        return True
    except Exception as e:
        print(f"Error ensuring Qdrant collection: {e}")
        return False

def create_email_embedding(text):
    """Create embedding for email text using OpenAI."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

def chunk_email_text(text, max_chunk_size=1000):
    """Split email text into chunks for better vector analysis."""
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def email_exists_in_qdrant(msg_id):
    """Check if an email with the given msg_id already exists in Qdrant."""
    try:
        # Ensure collection exists
        if not ensure_qdrant_collection():
            print("Failed to ensure Qdrant collection exists")
            return False
        
        # Search for existing email with this msg_id
        results = qdrant_client.scroll(
            collection_name="emails",
            limit=10
        )
        
        for point in results[0]:
            if point.payload.get('msg_id') == msg_id and point.payload.get('type') == 'full_email':
                return True
        
        return False
        
    except Exception as e:
        print(f"Error checking if email exists in Qdrant: {e}")
        return False

def store_email_in_qdrant(email_data):
    """Store email data in Qdrant with vector embeddings."""
    try:
        # Check if email already exists
        if email_exists_in_qdrant(email_data['msg_id']):
            print(f"Email {email_data['msg_id']} already exists in Qdrant, skipping...")
            return True
        
        # Ensure collection exists
        if not ensure_qdrant_collection():
            print("Failed to ensure Qdrant collection exists")
            return False
        
        msg_id = email_data['msg_id']
        email_text = f"Subject: {email_data['subject']}\nFrom: {email_data['from_email']}\nBody: {email_data['body']}"
        
        # Create embedding for the full email
        embedding = create_email_embedding(email_text)
        if not embedding:
            print(f"Failed to create embedding for email {msg_id}")
            return False
        
        # Store the full email
        qdrant_client.upsert(
            collection_name="emails",
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'msg_id': msg_id,
                        'received': email_data['received'],
                        'from_email': email_data['from_email'],
                        'subject': email_data['subject'],
                        'body': email_data['body'],
                        'processed': None,
                        'type': 'full_email'
                    }
                )
            ]
        )
        
        # Also store individual chunks for better analysis
        chunks = chunk_email_text(email_data['body'])
        for i, chunk in enumerate(chunks):
            chunk_embedding = create_email_embedding(chunk)
            if chunk_embedding:
                qdrant_client.upsert(
                    collection_name="emails",
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=chunk_embedding,
                            payload={
                                'msg_id': msg_id,
                                'chunk_index': i,
                                'chunk_text': chunk,
                                'subject': email_data['subject'],
                                'from_email': email_data['from_email'],
                                'received': email_data['received'],
                                'processed': None,
                                'type': 'email_chunk'
                            }
                        )
                    ]
                )
        
        print(f"Successfully stored email {msg_id} in Qdrant")
        return True
        
    except Exception as e:
        print(f"Error storing email in Qdrant: {e}")
        return False

def get_unprocessed_emails_from_qdrant():
    """Get unprocessed emails from Qdrant."""
    try:
        # Ensure collection exists
        if not ensure_qdrant_collection():
            print("Failed to ensure Qdrant collection exists")
            return []
        
        # Get all points and filter in Python to avoid index issues
        results = qdrant_client.scroll(
            collection_name="emails",
            limit=1000
        )
        
        emails = []
        for point in results[0]:
            # Filter for full emails that are not processed
            if (point.payload.get('type') == 'full_email' and 
                point.payload.get('processed') is None):
                emails.append({
                    'msg_id': point.payload['msg_id'],
                    'received': point.payload['received'],
                    'from_email': point.payload['from_email'],
                    'subject': point.payload['subject'],
                    'body': point.payload['body'],
                    'point_id': point.id
                })
        
        return emails
        
    except Exception as e:
        print(f"Error getting unprocessed emails from Qdrant: {e}")
        return []

def mark_email_as_processed_in_qdrant(point_id):
    """Mark an email as processed in Qdrant."""
    try:
        # Ensure collection exists
        if not ensure_qdrant_collection():
            print("Failed to ensure Qdrant collection exists")
            return False
        
        qdrant_client.set_payload(
            collection_name="emails",
            payload={'processed': int(time.time())},
            points=[point_id]
        )
        return True
    except Exception as e:
        print(f"Error marking email as processed in Qdrant: {e}")
        return False

def get_all_emails_from_qdrant():
    """Get all emails from Qdrant for display."""
    try:
        # Ensure collection exists
        if not ensure_qdrant_collection():
            print("Failed to ensure Qdrant collection exists")
            return []
        
        # Get all points and filter in Python to avoid index issues
        results = qdrant_client.scroll(
            collection_name="emails",
            limit=1000
        )
        
        emails = []
        for point in results[0]:
            # Only include full emails (not chunks)
            if point.payload.get('type') == 'full_email':
                emails.append({
                    'msg_id': point.payload['msg_id'],
                    'received': point.payload['received'],
                    'from_email': point.payload['from_email'],
                    'subject': point.payload['subject'],
                    'body': point.payload['body'],
                    'processed': point.payload.get('processed')
                })
        
        return emails
        
    except Exception as e:
        print(f"Error getting emails from Qdrant: {e}")
        return []

# ============================================================================
# AI & EVENT PROCESSING FUNCTIONS
# ============================================================================

def extract_events_with_ai(plain_text):
    """Extract events from email text using OpenAI."""
    try:
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": """You are an event extraction assistant. Extract event details from the given text and return them in a JSON array format.
Each event should be an object with these exact fields:
{
    "Event Name": "string",
    "Date": "string",
    "Start Time": "string",
    "End Time": "string",
    "City": "string",
    "State": "string",
    "Venue": "string",
    "Address": "string",
    "Description": "string",
    "URL": "string"
}
Only report clean start and end times including AM/PM but not timezones. Report State as a 2 letter abbreviation. Report date as MM/DD/YYYY of the event itself (not the email date). If year is unknown, assume event is in 2025. The word Location is not a Venue or Address. Return ONLY the JSON array, no other text. If no events are found, return an empty array [].

IMPORTANT: Look for any mention of events, gatherings, meetings, or activities in the text. Even if the information is incomplete, extract what you can find. If you see a date and time mentioned, it's likely an event. If you see a location mentioned, it's likely a venue. Extract as much information as possible, even if some fields are empty."""},
                {"role": "user", "content": plain_text}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        headers = {
            "Authorization": f"Bearer {my_secrets.openai_by}",
            "Content-Type": "application/json"
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print(f"OpenAI API error: {response.status_code}")
            return []
            
        data = response.json()
        
        if 'choices' not in data or not data['choices']:
            print("No choices in OpenAI response")
            return []
            
        content = data['choices'][0]['message']['content'].strip()
        if not content:
            print("Empty content in OpenAI response")
            return []
        
        try:
            # First try direct JSON parsing
            try:
                events = json.loads(content)
                if isinstance(events, list):
                    processed_events = process_event_data(events)
                    return processed_events
            except json.JSONDecodeError:
                print("Direct JSON parsing failed, trying to extract JSON array...")
                
            # If direct parsing fails, try to extract and clean the JSON array
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx + 1]
                
                # Fix truncated JSON by completing any incomplete objects
                parts = json_str.split('},')
                fixed_parts = []
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:  # Not the last part
                        if not part.strip().endswith('}'):
                            part = part + '}'
                    fixed_parts.append(part)
                json_str = '}'.join(fixed_parts)
                
                # Remove duplicate fields by keeping the last occurrence
                for field in ["URL", "Event Name"]:
                    pattern = f'"{field}":[^,}}]+,\\s*"{field}":'
                    while re.search(pattern, json_str):
                        match = re.search(pattern, json_str)
                        if match:
                            start, end = match.span()
                            second_field_start = json_str.find(f'"{field}":', start + len(field) + 4)
                            json_str = json_str[:start] + json_str[second_field_start:]
                
                # Fix any remaining JSON syntax issues
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                
                try:
                    events = json.loads(json_str)
                    if not isinstance(events, list):
                        print("Response is not a list")
                        return []
                    processed_events = process_event_data(events)
                    return processed_events
                except json.JSONDecodeError as e:
                    print(f"Error parsing cleaned JSON: {e}")
                    return []
            else:
                print("No JSON array found in response")
                return []
                
        except Exception as e:
            print(f"Error processing response: {e}")
            return []
            
    except Exception as e:
        print(f"Error in extract_events_with_ai: {e}")
        return []

def process_event_data(events, sanitize=True):
    """Process and clean event data to ensure all required fields exist and are properly formatted."""
    processed_events = []
    for event in events:
        # Ensure all required fields exist
        processed_event = {
            "Event Name": event.get("Event Name", ""),
            "Date": event.get("Date", ""),
            "Start Time": event.get("Start Time", ""),
            "End Time": event.get("End Time", ""),
            "City": event.get("City", ""),
            "State": event.get("State", ""),
            "Venue": event.get("Venue", ""),
            "Address": event.get("Address", ""),
            "Description": event.get("Description", ""),
            "URL": event.get("URL", "")
        }
        
        if sanitize:
            # Sanitize event data to remove problematic characters
            for key, value in processed_event.items():
                if isinstance(value, str):
                    processed_event[key] = value.replace('::', '-').replace(':', '-')
        
        processed_events.append(processed_event)
    return processed_events

def fetch_missing_data(url, missing_fields):
    """Fetch missing data from event URL using OpenAI with rate limiting."""
    try:
        # Sanitize URL first
        url = sanitize_url(url)
        if not url:
            print(f"Invalid URL: {url}")
            return {}
            
        # Fetch website content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all text from the webpage
        website_text = ' '.join([text.strip() for text in soup.stripped_strings])
        
        # Use OpenAI to analyze the website text
        missing_fields_str = ", ".join(missing_fields)
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": f"""You are an event data extraction assistant. Analyze the provided text and extract ONLY the following missing fields: {missing_fields_str}. Only report clean start and end times, not timezones. Report State as a 2 letter abbreviation, and the word Location is not a Venue or Address. Report date as MM/DD/YYYY. Return a JSON object with ONLY these fields. Return ONLY the JSON object, no other text."""},
                {"role": "user", "content": website_text}
            ],
            "temperature": 0.3
        }
        
        headers = {
            "Authorization": f"Bearer {my_secrets.openai_by}",
            "Content-Type": "application/json"
        }
        
        # Add delay to respect rate limits (3 calls per minute)
        time.sleep(20)  # Wait 20 seconds between calls
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if 'choices' not in data or not data['choices']:
            print("No choices in OpenAI response")
            return {}
            
        content = data['choices'][0]['message']['content'].strip()
        if not content:
            print("Empty content in OpenAI response")
            return {}
            
        try:
            # Try to find JSON object in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx + 1]
                event_data = json.loads(json_str)
                return event_data
            else:
                print("No JSON object found in response")
                return {}
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return {}
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from URL {url}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error fetching data from URL {url}: {e}")
        return {} 

# ============================================================================
# DATA STORAGE FUNCTIONS
# ============================================================================

def write_to_dynamo(table_name, data):
    """Write data to specified DynamoDB table."""
    try:
        if table_name == "events":
            # Generate a unique event ID
            event_id = f"{data[0]}_{int(time.time())}"
            
            # Convert date string to timestamp if it's in the correct format
            try:
                event_date = process_datetime(data[1], "timestamp")
                # Check if event date is in the past
                if event_date < int(time.time()):
                    print(f"Skipping past event: {data[0]} on {data[1]} (timestamp: {event_date})")
                    return
            except Exception as e:
                print(f"Failed to parse date '{data[1]}': {e}")
                event_date = int(time.time())  # Use current time if date parsing fails
            
            item = {
                'event_id': event_id,
                'event_name': data[0],
                'date': event_date,
                'start_time': data[2],
                'end_time': data[3],
                'city': data[4],
                'state': data[5],
                'venue': data[6],
                'address': data[7],
                'description': data[8],
                'url': data[9]
            }
            
            events_table.put_item(Item=item)
            print(f"Successfully wrote event to DynamoDB")
            
        else:
            print(f"Unknown table: {table_name}")
            
    except Exception as e:
        print(f"Error writing to DynamoDB table {table_name}: {e}")
        if table_name == "events":
            raise

def get_events():
    """Get all events from DynamoDB."""
    try:
        print("Fetching events from DynamoDB...")
        response = events_table.scan()
        items = response.get('Items', [])
        print(f"Found {len(items)} events in DynamoDB")
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        if not df.empty:
            # Convert timestamps and rename columns
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'].fillna(0).astype(int), unit='s')
            # Rename columns to match expected format
            df = df.rename(columns={
                'event_name': 'Event Name',
                'date': 'Date',
                'start_time': 'Start Time',
                'end_time': 'End Time',
                'city': 'City',
                'state': 'State',
                'venue': 'Venue',
                'address': 'Address',
                'description': 'Description',
                'url': 'URL'
            })
            # Fill NaN values with empty strings
            df = df.fillna('')
            # Sort by date
            df = df.sort_values('Date', ascending=True)
            # Select only the display columns
            display_columns = ['Event Name', 'Date', 'Start Time', 'End Time', 'City', 'State', 'Venue', 'Address', 'Description', 'URL']
            df = df[display_columns]
        return df
    except Exception as e:
        print(f"Error getting events: {e}")
        return pd.DataFrame()

def get_event_emails():
    """Get all event emails from Qdrant."""
    try:
        emails = get_all_emails_from_qdrant()
        
        # Convert to DataFrame
        df = pd.DataFrame(emails)
        if not df.empty:
            # Convert timestamps to readable dates, handling None values
            if 'received' in df.columns:
                df['received'] = pd.to_datetime(df['received'].fillna(0).astype(int), unit='s')
            if 'processed' in df.columns:
                df['processed'] = pd.to_datetime(df['processed'].fillna(0).astype(int), unit='s')
        return df
    except Exception as e:
        st.error(f"Error getting event emails: {e}")
        return pd.DataFrame()

def update_existing_event(tool, event_data, existing_data, **kwargs):
    """Update existing event with missing data in specified tool."""
    try:
        if tool == "sheets":
            service = kwargs.get('service')
            spreadsheet_id = kwargs.get('spreadsheet_id')
            existing_row = kwargs.get('existing_row')
            
            if not all([service, spreadsheet_id, existing_row]):
                print("Missing required parameters for sheets update")
                return
                
            # Find the row number of the existing event
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range="Events!A:J"
            ).execute()
            values = result.get('values', [])
            
            # Find the row number by matching the URL
            row_number = None
            for idx, row in enumerate(values[1:], start=2):  # Skip header row
                if len(row) > 9 and row[9] == event_data[9]:  # Compare URLs
                    row_number = idx
                    break
            
            if row_number is None:
                print(f"Could not find row number for URL: {event_data[9]}")
                return
                
            updated_values = []
            for i, (new_val, existing_val) in enumerate(zip(event_data, existing_row)):
                if not existing_val and new_val:
                    updated_values.append(new_val)
                else:
                    updated_values.append(existing_val)
            
            # Update the row with new values using the row number
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f"Events!A{row_number}:J{row_number}",
                valueInputOption="RAW",
                body={"values": [updated_values]}
            ).execute()
            
        elif tool == "dynamo":
            existing_event = kwargs.get('existing_event')
            
            if not existing_event:
                print("Missing existing_event parameter for dynamo update")
                return
                
            update_expressions = []
            expression_values = {}
            
            # Check each field and add to update if missing in existing event
            fields = {
                'event_name': 0,
                'date': 1,
                'start_time': 2,
                'end_time': 3,
                'city': 4,
                'state': 5,
                'venue': 6,
                'address': 7,
                'description': 8,
                'url': 9
            }
            
            for field, index in fields.items():
                if not existing_event.get(field) and event_data[index]:
                    update_expressions.append(f"{field} = :{field}")
                    expression_values[f":{field}"] = event_data[index]
            
            if update_expressions:
                events_table.update_item(
                    Key={'event_id': existing_event['event_id']},
                    UpdateExpression='SET ' + ', '.join(update_expressions),
                    ExpressionAttributeValues=expression_values
                )
        else:
            print(f"Unknown tool: {tool}")
            
    except Exception as e:
        print(f"Error updating existing event in {tool}: {e}") 

# ============================================================================
# GOOGLE SHEETS FUNCTIONS
# ============================================================================

def write_to_sheet(email_data):
    """Write email data to Google Sheets."""
    service = get_service("sheets")
    if not service:
        return
    sheet = service.spreadsheets()

    SPREADSHEET_ID = my_secrets.SPREADSHEET_ID
    RANGE = 'Emails!A:F'
    
    # Get the last row to append after it
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE
    ).execute()
    values = result.get('values', [])
    last_row = len(values) if values else 0
    
    # Convert email data to list format and truncate email body text
    truncated_email_data = []
    for email in email_data:
        # Convert to list format: [msg_id, formatted_date, from_email, subject, body]
        formatted_date = process_datetime(email['received'], "formatted")
        row = [email['msg_id'], formatted_date, email['from_email'], email['subject'], email['body']]
        
        # Truncate body text if needed
        if len(row) > 4:  # If there's a body text (column E)
            text = row[4]
            if text and len(text) > GOOGLE_SHEETS_CHAR_LIMIT:
                row[4] = text[:GOOGLE_SHEETS_TRUNCATE_LIMIT] + TRUNCATE_SUFFIX
        truncated_email_data.append(row)
    
    # Write new data after the last row
    if truncated_email_data:
        # Add empty column F for processed status
        email_data_with_processed = [row + [""] for row in truncated_email_data]
        body = {'values': email_data_with_processed}
        sheet.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=f'Emails!A{last_row + 1}:F',
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body=body
        ).execute()

def export_to_google_sheets():
    """Export events from DynamoDB to Google Sheets."""
    try:
        # Verify Google Sheets access first
        if not check_config("google_sheets"):
            st.error("Unable to access Google Sheets. Please check your credentials and try again.")
            return False
            
        # Get events from DynamoDB
        events_df = get_events()
        if events_df.empty:
            st.warning("No events found to export.")
            return False
        
        # Convert Timestamp objects to strings
        if 'Date' in events_df.columns:
            events_df['Date'] = events_df['Date'].dt.strftime('%m/%d/%Y')
        
        # Convert DataFrame to list of lists for Google Sheets
        events_data = [events_df.columns.tolist()]  # Headers
        events_data.extend(events_df.values.tolist())
        
        # Get service and write data
        service = get_service("sheets")
        if not service:
            st.error("Unable to get Google Sheets service.")
            return False
        sheet = service.spreadsheets()
        
        try:
            # Clear existing data in the Events sheet
            sheet.values().clear(
                spreadsheetId=my_secrets.SPREADSHEET_ID,
                range="Events!A:J"
            ).execute()
            
            # Write new data
            sheet.values().update(
                spreadsheetId=my_secrets.SPREADSHEET_ID,
                range="Events!A1",
                valueInputOption="RAW",
                body={"values": events_data}
            ).execute()
            
            return True
        except Exception as e:
            print(f"Error in Google Sheets operation: {e}")
            st.error(f"Error in Google Sheets operation: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error exporting to Google Sheets: {e}")
        st.error(f"Error exporting to Google Sheets: {str(e)}")
        return False 

# ============================================================================
# EMAIL PROCESSING FUNCTIONS
# ============================================================================

def mark_as_read_and_archive(service, msg_id):
    """Mark email as read and archive it."""
    try:
        # Mark as read
        service.users().messages().modify(
            userId='me',
            id=msg_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        
        # Archive (remove INBOX label)
        service.users().messages().modify(
            userId='me',
            id=msg_id,
            body={'removeLabelIds': ['INBOX']}
        ).execute()
        return True
    except Exception as e:
        print(f"Error marking email as read and archived: {e}")
        return False

def process_emails(action="process_existing"):
    """
    Comprehensive email processing function.
    
    Args:
        action: "fetch_new" to get emails from Gmail, "process_existing" to process stored emails
    """
    try:
        if action == "fetch_new":
            # Fetch new emails from Gmail
            service = get_service("gmail")
            if not service:
                return [], 0
            
            # Get all messages from inbox
            results = service.users().messages().list(
                userId='me',
                labelIds=['INBOX']
            ).execute()
            messages = results.get('messages', [])
            
            email_data = []
            processed_count = 0
            
            for msg in messages:
                msg_id = msg['id']
                msg = service.users().messages().get(userId='me', id=msg_id).execute()
                payload = msg['payload']
                headers = payload['headers']

                from_email = subject = date = body = None
                
                for header in headers:
                    if header['name'] == 'From':
                        from_email = header['value']
                    if header['name'] == 'Subject':
                        subject = header['value']
                    if header['name'] == 'Date':
                        date = header['value']

                # Extract email body
                if 'parts' in payload:
                    for part in payload['parts']:
                        if part['mimeType'] == 'text/plain':
                            if 'data' in part['body']:
                                body = base64.urlsafe_b64decode(part['body']['data']).decode()
                                break
                        elif part['mimeType'] == 'multipart/alternative':
                            # Handle nested parts
                            for subpart in part['parts']:
                                if subpart['mimeType'] == 'text/plain':
                                    if 'data' in subpart['body']:
                                        body = base64.urlsafe_b64decode(subpart['body']['data']).decode()
                                        break
                elif 'body' in payload and 'data' in payload['body']:
                    body = base64.urlsafe_b64decode(payload['body']['data']).decode()

                if from_email and subject and date and body:
                    # Format the date
                    formatted_date = process_datetime(date, "formatted")
                    email_data.append({
                        'msg_id': msg_id,
                        'received': process_datetime(date, "timestamp"),
                        'from_email': from_email,
                        'subject': subject,
                        'body': body
                    })
                    
                    # Mark as read and archive
                    try:
                        mark_as_read_and_archive(service, msg_id)
                        processed_count += 1
                    except Exception as e:
                        print(f"Error marking email as read and archived: {e}")

            # Store fetched emails in Qdrant
            for email in email_data:
                store_email_in_qdrant(email)
            
            return email_data, processed_count
            
        elif action == "process_existing":
            # Read unprocessed emails from Qdrant
            unprocessed_emails = get_unprocessed_emails_from_qdrant()
            
            if not unprocessed_emails:
                print("No unprocessed emails found")
                return
            
            print(f"Found {len(unprocessed_emails)} unprocessed emails")
            
            for email in unprocessed_emails:
                try:
                    print(f"Processing email: {email['subject']}")
                    
                    # Clean the email body text
                    body_text = email['body']
                    if not body_text:
                        print(f"Skipping email {email['subject']} - no valid text content")
                        continue
                    
                    # First, extract and temporarily replace URLs with placeholders
                    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                    urls = re.findall(url_pattern, body_text)
                    
                    # Create a mapping of placeholders to URLs
                    url_placeholders = {}
                    for i, url in enumerate(urls):
                        placeholder = f"__URL_{i}__"
                        url_placeholders[placeholder] = url
                        body_text = body_text.replace(url, placeholder)
                    
                    # Remove HTML tags but preserve their content
                    body_text = re.sub(r'<[^>]+>', ' ', body_text)
                    
                    # Remove email addresses
                    body_text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', body_text)
                    
                    # Clean up extra whitespace and normalize spaces
                    body_text = re.sub(r'\s+', ' ', body_text)
                    
                    # Restore URLs from placeholders
                    for placeholder, url in url_placeholders.items():
                        body_text = body_text.replace(placeholder, url)
                    
                    # Final cleanup - be very careful not to break URLs
                    # Only remove truly problematic characters, preserve URL characters
                    body_text = re.sub(r'[^\w\s.,!?\-/:@#$%&*()+=]', ' ', body_text)
                    body_text = re.sub(r'\s+', ' ', body_text)
                    
                    # Truncate to approximately 6000 tokens (assuming 4 chars per token)
                    max_length = 24000  # 6000 tokens * 4 chars per token
                    if len(body_text) > max_length:
                        body_text = body_text[:max_length] + "... (truncated)"
                    cleaned_body = body_text.strip()
                    
                    if not cleaned_body:
                        print(f"Skipping email {email['subject']} - no valid text content")
                        continue
                    
                    # Extract events from the cleaned email body
                    events = extract_events_with_ai(cleaned_body)
                    print(f"Extracted {len(events)} events from email")
                    
                    if events:
                        for event in events:
                            try:
                                print(f"Processing event: {event.get('Event Name', 'Unnamed Event')}")
                                # Format the event data
                                event_data = [
                                    event.get("Event Name", ""),
                                    event.get("Date", ""),
                                    event.get("Start Time", ""),
                                    event.get("End Time", ""),
                                    event.get("City", ""),
                                    event.get("State", ""),
                                    event.get("Venue", ""),
                                    event.get("Address", ""),
                                    event.get("Description", ""),
                                    sanitize_url(event.get("URL", ""))
                                ]
                                
                                # Write event to DynamoDB
                                write_to_dynamo("events", event_data)
                                
                            except Exception as e:
                                print(f"Error processing event: {e}")
                                continue
                        
                        # Mark as processed if we successfully extracted and stored events
                        try:
                            mark_email_as_processed_in_qdrant(email['point_id'])
                            print(f"Successfully processed email: {email['subject']}")
                        except Exception as e:
                            print(f"Error marking email as processed: {e}")
                    else:
                        print(f"No events found in email: {email['subject']}")
                        # Don't mark as processed if no events were found
                    
                except Exception as e:
                    print(f"Error processing email {email['subject']}: {e}")
                    # Don't mark as processed if there was an error
                    continue
                    
        else:
            print(f"Unknown action: {action}")
            
    except Exception as e:
        print(f"Error in process_emails: {e}")
        return [], 0 

# ============================================================================
# DATA MANAGEMENT FUNCTIONS
# ============================================================================

def clear_data(tool):
    """Clear data for specified tool."""
    try:
        if tool == "dynamo":
            print("Clearing DynamoDB data...")
            
            # Clear events table only (emails are now in Qdrant)
            print("Clearing events table...")
            response = events_table.scan()
            items = response.get('Items', [])
            cleared_events = 0
            for item in items:
                try:
                    events_table.delete_item(Key={'event_id': item['event_id']})
                    cleared_events += 1
                except Exception as e:
                    print(f"Error deleting event {item['event_id']}: {str(e)}")
                    continue
            
            print(f"Successfully cleared {cleared_events} events from DynamoDB")
            return True, 0, cleared_events
            
        elif tool == "qdrant":
            print("Clearing Qdrant cluster...")
            
            # Check if Qdrant credentials are configured
            if not hasattr(my_secrets, 'QDRANT_URL') or not hasattr(my_secrets, 'QDRANT_API_KEY'):
                print("Qdrant credentials not configured in my_secrets.py")
                return False, 0, 0
                
            # Initialize Qdrant client with credentials from my_secrets
            try:
                client = QdrantClient(url=my_secrets.QDRANT_URL, api_key=my_secrets.QDRANT_API_KEY)
            except Exception as e:
                print(f"Failed to connect to Qdrant: {str(e)}")
                return False, 0, 0
            
            # Test connection by getting collections
            try:
                collections = client.get_collections().collections
                if not collections:
                    print("No collections found in Qdrant cluster")
                    return True, 0, 0
            except Exception as e:
                print(f"Failed to get collections from Qdrant: {str(e)}")
                return False, 0, 0
            
            cleared_count = 0
            total_points_cleared = 0
            
            for collection in collections:
                collection_name = collection.name
                try:
                    print(f"Clearing collection: {collection_name}")
                    
                    # Get all points from the collection first
                    all_points = []
                    offset = None
                    
                    while True:
                        results = client.scroll(
                            collection_name=collection_name,
                            limit=100,
                            offset=offset
                        )
                        
                        points = results[0]
                        if not points:
                            break
                            
                        all_points.extend(points)
                        offset = results[1]  # Next page offset
                        
                        if not offset:
                            break
                    
                    if all_points:  # If there are points to delete
                        # Extract point IDs
                        point_ids = [point.id for point in all_points]
                        
                        # Delete points by their actual IDs
                        if point_ids:
                            client.delete(
                                collection_name=collection_name,
                                points_selector=models.PointIdsList(
                                    points=point_ids
                                )
                            )
                            print(f"Cleared {len(point_ids)} points from collection: {collection_name}")
                            total_points_cleared += len(point_ids)
                        else:
                            print(f"No points found in collection: {collection_name}")
                    else:
                        print(f"Collection {collection_name} is already empty")
                    
                    cleared_count += 1
                except Exception as e:
                    print(f"Failed to clear collection {collection_name}: {str(e)}")
                    continue
            
            if cleared_count > 0:
                print(f"Successfully cleared {cleared_count} collections with {total_points_cleared} total points")
                return True, cleared_count, total_points_cleared
            else:
                print("No collections were cleared")
                return False, 0, 0
                
        else:
            print(f"Unknown tool: {tool}")
            return False, 0, 0
            
    except Exception as e:
        print(f"Error clearing {tool} data: {str(e)}")
        return False, 0, 0 

# ============================================================================
# STREAMLIT INTERFACE & MAIN FUNCTION
# ============================================================================

def streamlit_interface():
    """Streamlit interface for the Event Agent."""
    st.title("Event Agent Dashboard")
    
    # 1. Database Configuration Section
    st.header("🗄️ Database Configuration")
    
    # Create columns for status and actions
    db_status_col, db_action_col = st.columns(2)
    
    with db_status_col:
        # DynamoDB Status
        dynamo_status = check_config("dynamo")
        st.metric("DynamoDB Tables", "✅ Ready" if dynamo_status else "❌ Missing")
        
        # AWS Status
        aws_status = check_config("aws")
        st.metric("AWS Access", "✅ Connected" if aws_status else "❌ Disconnected")
        
        # Qdrant Status
        qdrant_status = check_config("qdrant")
        if qdrant_status:
            st.metric("Qdrant Status", "✅ Connected")
        else:
            st.metric("Qdrant Status", "❌ Disconnected")
    
    with db_action_col:
        # Setup DynamoDB if needed
        if not dynamo_status:
            st.warning("DynamoDB events table is not set up.")
            if st.button("Setup DynamoDB Tables", use_container_width=True):
                with st.spinner("Setting up DynamoDB tables..."):
                    run_setup_script()
                    st.rerun()
        
        # Clear DynamoDB Data
        if st.button("🗑️ Clear DynamoDB Data", use_container_width=True):
            with st.spinner("Clearing DynamoDB data..."):
                success, _, events_cleared = clear_data("dynamo")
                if success:
                    st.success(f"DynamoDB cleared successfully! Removed {events_cleared} events.")
                    st.rerun()
                else:
                    st.error("Failed to clear DynamoDB data. Check the terminal logs for details.")
        
        # Clear Qdrant Cluster
        if st.button("🗑️ Clear Qdrant Cluster", use_container_width=True):
            with st.spinner("Clearing Qdrant cluster..."):
                if not qdrant_status:
                    st.warning("Qdrant is not configured.")
                else:
                    success, collections_cleared, points_cleared = clear_data("qdrant")
                    if success:
                        st.success(f"Qdrant cluster cleared successfully! Cleared {collections_cleared} collections with {points_cleared} total points.")
                    else:
                        st.error("Failed to clear Qdrant cluster. Check the terminal logs for details.")
    
    # 2. Gmail Configuration Section
    st.header("📧 Gmail Configuration")
    
    # Create columns for status and actions
    gmail_status_col, gmail_action_col = st.columns(2)
    
    with gmail_status_col:
        # Gmail Status
        gmail_status = check_config("gmail")
        st.metric("Gmail Access", "✅ Connected" if gmail_status else "❌ Disconnected")
        
        # If Gmail is disconnected, show setup button
        if not gmail_status:
            if st.button("🔄 Run Setup", use_container_width=True):
                with st.spinner("Running setup..."):
                    run_setup_script()
                    st.rerun()
    
    with gmail_action_col:
        # Check for New Emails
        if st.button("📥 Check for New Emails", use_container_width=True):
            with st.spinner("Fetching new emails from Gmail..."):
                email_data, processed_count = process_emails(action="fetch_new")
                if email_data:
                    print(f"Successfully fetched {processed_count} emails from Gmail and stored in Qdrant.")
                    st.success(f"Successfully fetched {processed_count} emails from Gmail and stored in Qdrant.")
                else:
                    print("No new emails found in inbox.")
                    st.info("No new emails found in inbox.")
        
        # Process Emails
        if st.button("⚙️ Process Emails", use_container_width=True):
            with st.spinner("Processing emails to extract events..."):
                # Then process emails
                process_emails()
                st.success("Email processing completed!")
    
    # 3. Data Access Section
    st.header("📊 Data Access")
    st.markdown("### 📋 Google Sheet")

    # Export to Google Sheets
    if st.button("📤 Export to Google Sheets", use_container_width=True):
        with st.spinner("Exporting events to Google Sheets..."):
            if export_to_google_sheets():
                st.success("Events exported to Google Sheets successfully!")
            else:
                st.error("Failed to export events to Google Sheets. Check logs for details.")

    # Google Sheet Link
    if st.button("📋 Open Google Sheet", use_container_width=True):
        st.markdown(f'<a href="https://docs.google.com/spreadsheets/d/' + my_secrets.SPREADSHEET_ID + '" target="_blank">Click here to open Google Sheet</a>', unsafe_allow_html=True)
        st.success("Opening Google Sheet in new tab...")
    
    # Data Tables
    st.markdown("### 📨 Event Emails")
    emails_df = get_event_emails()
    if not emails_df.empty:
        st.dataframe(emails_df, use_container_width=True)
    else:
        st.info("No event emails found.")
    
    st.markdown("### 🎯 Events")
    events_df = get_events()
    if not events_df.empty:
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input("Filter by date (optional)", value=None)
            if date_filter is not None:  # Only filter if a date is selected
                events_df = events_df[events_df['Date'].dt.date == date_filter]
        
        with col2:
            city_filter = st.text_input("Filter by city (optional)")
            if city_filter:
                events_df = events_df[events_df['City'].str.contains(city_filter, case=False, na=False)]
        
        # Display filtered events
        st.dataframe(events_df, use_container_width=True)
    else:
        st.info("No events found.")

# Main function
if __name__ == "__main__":
    try:
        import streamlit as st
        streamlit_interface()
    except ImportError:
        # If not in Streamlit, run the original main function
        try:
            # First, get emails from Gmail
            email_data, processed_count = process_emails(action="fetch_new")
            if email_data:
                for email in email_data:
                    store_email_in_qdrant(email)
                print(f"Successfully fetched {processed_count} emails from Gmail and stored in Qdrant.")
            else:
                print("No new emails found in inbox.")
                
            # Then, process emails to extract events
            process_emails()
        except Exception as e:
            print(f"Error in main execution: {e}")
    except Exception as e:
        print(f"Error in Streamlit execution: {e}") 
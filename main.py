import streamlit as st
import sys
import os
import boto3
import pandas as pd
import subprocess
import requests
import time
import json
import re
import base64
import email.utils
from datetime import datetime
from bs4 import BeautifulSoup
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import Secrets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../secrets')))
import my_secrets

# Gmail API SCOPES
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/spreadsheets'  # Added for Google Sheets access
]

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'secrets')) #REPLACE WITH YOUR PATH
import my_secrets #REPLACE WITH YOUR SECRETS FILE NAME
# print("Available attributes in secrets:", dir(my_secrets)) #HELPS WITH DEBUGGING

# Load credentials for Gmail and Sheets APIs
def load_credentials():
    """Load or refresh Google API credentials."""
    creds = None
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            # Verify the credentials have all required scopes
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
            
            # Save the credentials
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            print("Credentials saved successfully")
        except Exception as e:
            print(f"Error in OAuth flow: {e}")
            raise Exception("Failed to obtain valid credentials. Please try again.")
    
    return creds

def format_datetime(date_str):
    # Parse the email date string
    parsed_date = email.utils.parsedate_tz(date_str)
    if parsed_date:
        # Convert to datetime object
        dt = datetime.fromtimestamp(email.utils.mktime_tz(parsed_date))
        # Format as MM/DD/YYYY HH:MM AM/PM
        return dt.strftime('%m/%d/%Y %I:%M %p')
    return date_str

def get_email_body(payload):
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                if 'data' in part['body']:
                    return base64.urlsafe_b64decode(part['body']['data']).decode()
            elif part['mimeType'] == 'multipart/alternative':
                # Handle nested parts
                for subpart in part['parts']:
                    if subpart['mimeType'] == 'text/plain':
                        if 'data' in subpart['body']:
                            return base64.urlsafe_b64decode(subpart['body']['data']).decode()
    elif 'body' in payload and 'data' in payload['body']:
        return base64.urlsafe_b64decode(payload['body']['data']).decode()
    return None

def mark_as_read_and_archive(service, msg_id):
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

# Extract email data from Gmail
def extract_emails():
    creds = load_credentials()
    service = build('gmail', 'v1', credentials=creds)
    
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

        body = get_email_body(payload)

        if from_email and subject and date and body:
            # Format the date
            formatted_date = format_datetime(date)
            email_data.append([msg_id, formatted_date, from_email, subject, body])
            
            # Mark as read and archive
            if mark_as_read_and_archive(service, msg_id):
                processed_count += 1

    return email_data, processed_count

def truncate_text(text, max_length=50000):
    """Truncate text to ensure it doesn't exceed Google Sheets' character limit of 50,000."""
    if text and len(text) > max_length:
        return text[:max_length - 15] + "... (truncated)"  # -15 to ensure room for truncation message
    return text

# Write to Google Sheets
def write_to_sheet(email_data):
    creds = load_credentials()
    service = build('sheets', 'v4', credentials=creds)
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
    
    # Truncate email body text before writing to sheet
    truncated_email_data = []
    for row in email_data:
        truncated_row = list(row)  # Create a copy of the row
        if len(row) > 4:  # If there's a body text (column E)
            truncated_row[4] = truncate_text(row[4])  # Truncate the body text
        truncated_email_data.append(truncated_row)
    
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

def sanitize_url(url):
    """Sanitize URL to ensure proper format."""
    if not url:
        return ""
    # Fix common URL issues
    url = url.replace('https-//', 'https://')
    url = url.replace('http-//', 'http://')
    # Remove any query parameters
    url = url.split('?')[0]
    return url

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
        API_KEY = my_secrets.openai_by
        ENDPOINT = "https://api.openai.com/v1/chat/completions"
        
        # Create a focused prompt based on missing fields
        missing_fields_str = ", ".join(missing_fields)
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": f"""You are an event data extraction assistant. Analyze the provided text and extract ONLY the following missing fields: {missing_fields_str}. Only report clean start and end times, not timezones. Report State as a 2 letter abbreviation. Report date as MM/DD/YYYY. The word Location is not a Venue or Address.
Return a JSON object with ONLY these fields. Return ONLY the JSON object, no other text."""},
                {"role": "user", "content": website_text}
            ],
            "temperature": 0.3
        }
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Add delay to respect rate limits (3 calls per minute)
        time.sleep(20)  # Wait 20 seconds between calls
        
        response = requests.post(ENDPOINT, headers=headers, json=payload)
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
            print(f"Raw response: {content}")
            return {}
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from URL {url}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error fetching data from URL {url}: {e}")
        return {}

def update_existing_event(service, spreadsheet_id, event_data, existing_row):
    """Update existing event with missing data."""
    try:
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
    except Exception as e:
        print(f"Error updating existing event: {e}")

# Extract events using OpenAI
def extract_events_with_ai(plain_text):
    """Extract events from email text using OpenAI."""
    try:
        # Process the entire email text without truncation
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
Only report clean start and end times, not timezones. Report State as a 2 letter abbreviation. Report date as MM/DD/YYYY. The word Location is not a Venue or Address. Return ONLY the JSON array, no other text. If no events are found, return an empty array [].

IMPORTANT: Look for any mention of events, gatherings, meetings, or activities in the text. Even if the information is incomplete, extract what you can find. If you see a date and time mentioned, it's likely an event. If you see a location mentioned, it's likely a venue. Extract as much information as possible, even if some fields are empty."""},
                {"role": "user", "content": plain_text}
            ],
            "temperature": 0.3,
            "max_tokens": 4000  # Increased token limit to handle longer emails
        }
        
        headers = {
            "Authorization": f"Bearer {my_secrets.openai_by}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if response.status_code != 200:
                print(f"OpenAI API error: {response.status_code}")
                print(f"Response content: {response.text}")
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
                        print(f"Successfully parsed {len(events)} events")
                        return clean_events(events)
                except json.JSONDecodeError:
                    print("Direct JSON parsing failed, trying to extract JSON array...")
                    
                # If direct parsing fails, try to extract and clean the JSON array
                start_idx = content.find('[')
                end_idx = content.rfind(']')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx + 1]
                    print("Extracted JSON string:", json_str)
                    
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
                    
                    print("Cleaned JSON string:", json_str)
                    
                    try:
                        events = json.loads(json_str)
                        if not isinstance(events, list):
                            print("Response is not a list")
                            return []
                        print(f"Successfully parsed {len(events)} events from cleaned JSON")
                        return clean_events(events)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing cleaned JSON: {e}")
                        print(f"Cleaned JSON string: {json_str}")
                        return []
                else:
                    print("No JSON array found in response")
                    return []
                    
            except Exception as e:
                print(f"Error processing response: {e}")
                print(f"Raw response: {content}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Error making OpenAI API request: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error in extract_events_with_ai: {e}")
            return []
            
    except Exception as e:
        print(f"Error in extract_events_with_ai: {e}")
        return []

def clean_events(events):
    """Clean up event data to ensure all required fields exist and are properly formatted."""
    cleaned_events = []
    for event in events:
        # Ensure all required fields exist
        cleaned_event = {
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
        cleaned_events.append(cleaned_event)
    return cleaned_events

def sanitize_event_data(event_data):
    """Sanitize event data to remove problematic characters."""
    sanitized = []
    for field in event_data:
        if isinstance(field, str):
            # Remove or replace problematic characters
            sanitized_field = field.replace('::', '-').replace(':', '-')
        else:
            sanitized_field = field
        sanitized.append(sanitized_field)
    return sanitized

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb',
    aws_access_key_id=my_secrets.event_agent_aws_access_key_id,
    aws_secret_access_key=my_secrets.event_agent_aws_secret_access_key,
    region_name=my_secrets.event_agent_aws_region
)

# Get DynamoDB tables
event_emails_table = dynamodb.Table('event_emails')
events_table = dynamodb.Table('events')

def datetime_to_timestamp(dt_str):
    """Convert datetime string to Unix timestamp."""
    try:
        dt = datetime.strptime(dt_str, '%m/%d/%Y %I:%M %p')
        return int(dt.timestamp())
    except:
        return int(time.time())  # Return current time if parsing fails

def write_to_dynamo(email_data):
    """Write email data to DynamoDB."""
    for row in email_data:
        msg_id, date_str, from_email, subject, body = row
        
        item = {
            'msg_id': msg_id,
            'received': datetime_to_timestamp(date_str),
            'sender': from_email,
            'subject': subject,
            'body': body,
            'processed': None  # Will be updated when processed
        }
        
        try:
            event_emails_table.put_item(Item=item)
        except Exception as e:
            print(f"Error writing to DynamoDB: {e}")

def mark_as_processed(msg_id):
    """Mark an email as processed in DynamoDB."""
    try:
        current_time = int(time.time())
        event_emails_table.update_item(
            Key={'msg_id': msg_id},
            UpdateExpression='SET #p = :val',
            ExpressionAttributeNames={
                '#p': 'processed'
            },
            ExpressionAttributeValues={
                ':val': current_time
            }
        )
    except Exception as e:
        st.error(f"Error marking email as processed: {e}")

def write_event_to_dynamo(event_data):
    """Write event data to DynamoDB."""
    try:
        # Generate a unique event ID
        event_id = f"{event_data[0]}_{int(time.time())}"
        
        # Convert date string to timestamp if it's in the correct format
        try:
            event_date = datetime_to_timestamp(event_data[1])
            # Check if event date is in the past
            if event_date < int(time.time()):
                print(f"Skipping past event: {event_data[0]} on {event_data[1]}")
                return
        except:
            event_date = int(time.time())  # Use current time if date parsing fails
        
        item = {
            'event_id': event_id,
            'event_name': event_data[0],
            'date': event_date,
            'start_time': event_data[2],
            'end_time': event_data[3],
            'city': event_data[4],
            'state': event_data[5],
            'venue': event_data[6],
            'address': event_data[7],
            'description': event_data[8],
            'url': event_data[9]
        }
        
        print(f"Writing event to DynamoDB: {item}")  # Debug log
        
        events_table.put_item(Item=item)
        print(f"Successfully wrote event to DynamoDB")  # Debug log
        
    except Exception as e:
        print(f"Error writing event to DynamoDB: {e}")  # Debug log
        raise

def clean_email_text(text):
    """Clean email text by removing HTML and formatting characters."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Truncate to approximately 6000 tokens (assuming 4 chars per token)
    max_length = 24000  # 6000 tokens * 4 chars per token
    if len(text) > max_length:
        text = text[:max_length] + "... (truncated)"
    
    return text.strip()

def process_emails():
    """Process unprocessed emails and extract events."""
    try:
        # Read unprocessed emails from DynamoDB
        response = event_emails_table.scan(
            FilterExpression='attribute_not_exists(#p)',
            ExpressionAttributeNames={
                '#p': 'processed'
            }
        )
        unprocessed_emails = response.get('Items', [])
        
        if not unprocessed_emails:
            print("No unprocessed emails found")
            return
        
        print(f"Found {len(unprocessed_emails)} unprocessed emails")
        
        for email in unprocessed_emails:
            try:
                print(f"Processing email: {email['subject']}")
                
                # Clean the email body text
                cleaned_body = clean_email_text(email['body'])
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
                            write_event_to_dynamo(event_data)
                            
                        except Exception as e:
                            print(f"Error processing event: {e}")
                            continue
                    
                    # Only mark as processed if we successfully extracted and stored events
                    mark_as_processed(email['msg_id'])
                    print(f"Successfully processed email: {email['subject']}")
                else:
                    print(f"No events found in email: {email['subject']}")
                    # Don't mark as processed if no events were found
                
            except Exception as e:
                print(f"Error processing email {email['subject']}: {e}")
                # Don't mark as processed if there was an error
                continue
                
    except Exception as e:
        print(f"Error in process_emails: {e}")
        pass

def update_existing_event(new_event_data, existing_event):
    """Update existing event with missing data."""
    try:
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
            if not existing_event.get(field) and new_event_data[index]:
                update_expressions.append(f"{field} = :{field}")
                expression_values[f":{field}"] = new_event_data[index]
        
        if update_expressions:
            events_table.update_item(
                Key={'event_id': existing_event['event_id']},
                UpdateExpression='SET ' + ', '.join(update_expressions),
                ExpressionAttributeValues=expression_values
            )
    except Exception as e:
        print(f"Error updating existing event: {e}")

def check_dynamo_tables():
    """Check if DynamoDB tables exist."""
    try:
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']
        return 'event_emails' in existing_tables and 'events' in existing_tables
    except Exception as e:
        st.error(f"Error checking DynamoDB tables: {e}")
        return False

def check_gmail_access():
    """Check if Gmail API access is working."""
    try:
        print("Attempting to get Gmail service...")  # Debug log
        service = get_gmail_service()
        if service:
            print("Gmail service obtained, testing connection...")  # Debug log
            service.users().labels().list(userId='me').execute()
            print("Gmail connection successful!")  # Debug log
            return True
        print("Failed to get Gmail service")  # Debug log
        return False
    except Exception as e:
        print(f"Error checking Gmail access: {str(e)}")  # Debug log
        st.error(f"Error checking Gmail access: {str(e)}")
        return False

def check_aws_access():
    """Check if AWS access is working."""
    try:
        dynamodb.meta.client.list_tables()
        return True
    except Exception as e:
        st.error(f"Error checking AWS access: {e}")
        return False

def run_setup_script():
    """Run the DynamoDB setup script."""
    try:
        result = subprocess.run(['python3', 'setup_dynamo.py'], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("DynamoDB tables created successfully!")
        else:
            st.error(f"Error creating tables: {result.stderr}")
    except Exception as e:
        st.error(f"Error running setup script: {e}")

def get_event_emails():
    """Get all event emails from DynamoDB."""
    try:
        response = event_emails_table.scan()
        items = response.get('Items', [])
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
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

def get_events():
    """Get all events from DynamoDB."""
    try:
        print("Fetching events from DynamoDB...")  # Debug log
        response = events_table.scan()
        items = response.get('Items', [])
        print(f"Found {len(items)} events in DynamoDB")  # Debug log
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        if not df.empty:
            print("Converting timestamps and renaming columns...")  # Debug log
            # Convert timestamps to readable dates, handling None values
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
            print(f"Processed DataFrame with {len(df)} rows")  # Debug log
            print("DataFrame columns:", df.columns.tolist())  # Debug log
            print("DataFrame sample:", df.head())  # Debug log
            print("DataFrame info:", df.info())  # Debug log
        return df
    except Exception as e:
        print(f"Error getting events: {e}")  # Debug log
        return pd.DataFrame()

def get_gmail_service():
    """Get Gmail API service instance."""
    try:
        print("Loading credentials from token.json...")  # Debug log
        # Load credentials from the token file
        creds = None
        if os.path.exists('token.json'):
            print("token.json exists, attempting to load credentials...")  # Debug log
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            print("Credentials loaded from token.json")  # Debug log
        else:
            print("token.json not found")  # Debug log
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            print("Credentials not valid, attempting to refresh or get new ones...")  # Debug log
            if creds and creds.expired and creds.refresh_token:
                print("Refreshing expired credentials...")  # Debug log
                creds.refresh(Request())
            else:
                print("No valid credentials found, starting OAuth flow...")  # Debug log
                if not os.path.exists('credentials.json'):
                    print("credentials.json not found!")  # Debug log
                    return None
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            print("Saving credentials to token.json...")  # Debug log
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        # Build the Gmail service
        print("Building Gmail service...")  # Debug log
        service = build('gmail', 'v1', credentials=creds)
        print("Gmail service built successfully")  # Debug log
        return service
    except Exception as e:
        print(f"Error getting Gmail service: {str(e)}")  # Debug log
        return None

def clear_processed_timestamps():
    """Clear the processed timestamp from all emails in DynamoDB."""
    try:
        # Get all emails
        response = event_emails_table.scan()
        items = response.get('Items', [])
        
        if not items:
            st.info("No emails found in the database.")
            return 0
            
        cleared_count = 0
        for item in items:
            try:
                # Remove the processed attribute
                event_emails_table.update_item(
                    Key={'msg_id': item['msg_id']},
                    UpdateExpression='REMOVE #p',
                    ExpressionAttributeNames={
                        '#p': 'processed'
                    }
                )
                cleared_count += 1
            except Exception as e:
                st.error(f"Error clearing processed timestamp for email {item['msg_id']}: {e}")
                continue
                
        return cleared_count
    except Exception as e:
        st.error(f"Error in clear_processed_timestamps: {e}")
        return 0

def clear_qdrant_cluster():
    """Clear all points from the Qdrant cluster while preserving collection schema."""
    try:
        # Check if Qdrant credentials are configured
        if not hasattr(my_secrets, 'QDRANT_URL') or not hasattr(my_secrets, 'QDRANT_API_KEY'):
            print("Qdrant credentials not configured in my_secrets.py")
            return False
        # Initialize Qdrant client with credentials from my_secrets
        try:
            client = QdrantClient(
                url=my_secrets.QDRANT_URL,
                api_key=my_secrets.QDRANT_API_KEY
            )
        except Exception as e:
            print(f"Failed to connect to Qdrant: {str(e)}")
            return False
        
        # Get all collections
        collections = client.get_collections().collections
        
        for collection in collections:
            collection_name = collection.name
            # Delete all points from the collection
            client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchAny(any=[1])  # This will match all points
                        )
                    ]
                )
            )
            print(f"Cleared all points from collection: {collection_name}")
        
        return True
    except Exception as e:
        print(f"Error clearing Qdrant cluster: {e}")
        return False

def check_google_sheets_access():
    """Check if Google Sheets API access is working."""
    try:
        creds = load_credentials()
        service = build('sheets', 'v4', credentials=creds)
        # Try to access the spreadsheet
        service.spreadsheets().get(spreadsheetId=my_secrets.SPREADSHEET_ID).execute()
        return True
    except Exception as e:
        print(f"Error checking Google Sheets access: {e}")
        return False

def export_to_google_sheets():
    """Export events from DynamoDB to Google Sheets."""
    try:
        # Verify Google Sheets access first
        if not check_google_sheets_access():
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
        
        # Get credentials and build service
        creds = load_credentials()
        service = build('sheets', 'v4', credentials=creds)
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

def refresh_gmail_credentials():
    """Refresh Gmail API credentials by running the OAuth flow again."""
    try:
        print("Starting OAuth flow to refresh credentials...")
        if not os.path.exists('credentials.json'):
            st.error("credentials.json not found! Please ensure it exists in the project directory.")
            return False
            
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        
        # Save the new credentials
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
        print("New credentials saved successfully")
        return True
    except Exception as e:
        print(f"Error refreshing credentials: {str(e)}")
        st.error(f"Error refreshing credentials: {str(e)}")
        return False

def clear_dynamo_data():
    """Clear all data from DynamoDB tables."""
    try:
        print("Clearing DynamoDB data...")
        
        # Clear event_emails table
        print("Clearing event_emails table...")
        response = event_emails_table.scan()
        items = response.get('Items', [])
        cleared_emails = 0
        for item in items:
            try:
                event_emails_table.delete_item(Key={'msg_id': item['msg_id']})
                cleared_emails += 1
            except Exception as e:
                print(f"Error deleting email {item['msg_id']}: {str(e)}")
                continue
        
        # Clear events table
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
        
        print(f"Successfully cleared {cleared_emails} emails and {cleared_events} events from DynamoDB")
        return True, cleared_emails, cleared_events
        
    except Exception as e:
        print(f"Error clearing DynamoDB data: {str(e)}")
        return False, 0, 0

def streamlit_interface():
    """Streamlit interface for the Event Agent."""
    st.title("Event Agent Dashboard")
    
    # Create two main columns for status and actions
    status_col, action_col = st.columns(2)
    
    # System Status Column
    with status_col:
        st.header("🔍 System Status")
        
        # Gmail Status
        gmail_status = check_gmail_access()
        st.metric("Gmail Access", "✅ Connected" if gmail_status else "❌ Disconnected")
        
        # If Gmail is disconnected, show refresh button
        if not gmail_status:
            if st.button("🔄 Refresh Gmail Credentials", use_container_width=True):
                with st.spinner("Refreshing Gmail credentials..."):
                    if refresh_gmail_credentials():
                        st.success("Credentials refreshed successfully! Please refresh the page.")
                        st.rerun()
                    else:
                        st.error("Failed to refresh credentials. Check the logs for details.")
        
        # Qdrant Status - Check if credentials are configured
        qdrant_configured = hasattr(my_secrets, 'QDRANT_URL') and hasattr(my_secrets, 'QDRANT_API_KEY')
        if qdrant_configured:
            st.metric("Qdrant Status", "✅ Configured")
        else:
            st.metric("Qdrant Status", "⚠️ Not Configured")
        
        # AWS Status
        aws_status = check_aws_access()
        st.metric("AWS Access", "✅ Connected" if aws_status else "❌ Disconnected")
        
        # DynamoDB Status
        dynamo_status = check_dynamo_tables()
        st.metric("DynamoDB Tables", "✅ Ready" if dynamo_status else "❌ Missing")
    
    # System Actions Column
    with action_col:
        st.header("⚡ System Actions")
        
        # Setup DynamoDB if needed
        if not dynamo_status:
            st.warning("DynamoDB tables are not set up. Click the button below to create them.")
            if st.button("Setup DynamoDB Tables", use_container_width=True):
                with st.spinner("Setting up DynamoDB tables..."):
                    run_setup_script()
                    st.rerun()
        
        # Check for New Emails
        if st.button("Check for New Emails", use_container_width=True):
            with st.spinner("Fetching new emails from Gmail..."):
                email_data, _ = extract_emails()
                if email_data:
                    write_to_dynamo(email_data)
        
        # Clear Qdrant Cluster
        if st.button("Clear Qdrant Cluster", use_container_width=True):
            with st.spinner("Clearing Qdrant cluster..."):
                if not qdrant_configured:
                    st.warning("Qdrant is not configured. Please add QDRANT_URL and QDRANT_API_KEY to my_secrets.py")
                else:
                    result = clear_qdrant_cluster()
                    if result:
                        st.success("Qdrant cluster cleared successfully!")
                    else:
                        st.error("Failed to clear Qdrant cluster. Check the terminal logs for details.")
        
        # Clear DynamoDB Data
        if st.button("🗑️ Clear DynamoDB Data", use_container_width=True):
            with st.spinner("Clearing DynamoDB data..."):
                success, emails_cleared, events_cleared = clear_dynamo_data()
                if success:
                    st.success(f"DynamoDB cleared successfully! Removed {emails_cleared} emails and {events_cleared} events.")
                    st.rerun()
                else:
                    st.error("Failed to clear DynamoDB data. Check the terminal logs for details.")
        
        # Clear Processed Timestamps
        if st.button("Clear Processed Timestamps", use_container_width=True):
            clear_processed_timestamps()
        
        # Reprocess Emails
        if st.button("Reprocess Emails", use_container_width=True):
            process_emails()
            
        # Export to Google Sheets
        if st.button("Export to Google Sheets", use_container_width=True):
            with st.spinner("Exporting events to Google Sheets..."):
                if export_to_google_sheets():
                    st.success("Events exported to Google Sheets successfully!")
                else:
                    st.error("Failed to export events to Google Sheets. Check logs for details.")
    
    # Display event emails
    st.header("📨 Event Emails")
    emails_df = get_event_emails()
    if not emails_df.empty:
        st.dataframe(emails_df, use_container_width=True)
    else:
        st.info("No event emails found.")
    
    # Display events
    st.header("🎯 Events")
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
            email_data, processed_count = extract_emails()
            if email_data:
                write_to_dynamo(email_data)
                print(f"Successfully fetched {processed_count} emails from Gmail and written to DynamoDB.")
            else:
                print("No new emails found in inbox.")
                
            # Then, process emails to extract events
            process_emails()
        except Exception as e:
            print(f"Error in main execution: {e}")
    except Exception as e:
        print(f"Error in Streamlit execution: {e}")
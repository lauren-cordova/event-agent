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
# print("Available attributes in secrets:", dir(my_secrets)) #HELPS WITH DEBUGGING

# Constants
GOOGLE_SHEETS_CHAR_LIMIT = 50000
GOOGLE_SHEETS_TRUNCATE_LIMIT = 49985
TRUNCATE_SUFFIX = "... (truncated)"

# Gmail API SCOPES
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/spreadsheets'  # Added for Google Sheets access
]

# Load or refresh Google API credentials
def load_credentials():
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

# Write to Google Sheets
def write_to_sheet(email_data):
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

# Sanitize URL to ensure proper format
def sanitize_url(url):
    if not url:
        return ""
    # Fix common URL issues
    url = url.replace('https-//', 'https://')
    url = url.replace('http-//', 'http://')
    # Remove any query parameters
    url = url.split('?')[0]
    return url

# Fetch missing data from event URL using OpenAI with rate limiting
def fetch_missing_data(url, missing_fields):
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
Only report clean start and end times including AM/PM but not timezones. Report State as a 2 letter abbreviation. Report date as MM/DD/YYYY of the event itself (not the email date). If year is unknown, assume event is current year. The word Location is not a Venue or Address. Return ONLY the JSON array, no other text. If no events are found, return an empty array [].

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
                        return process_event_data(events)
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
                        return process_event_data(events)
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
                    # Remove or replace problematic characters
                    processed_event[key] = value.replace('::', '-').replace(':', '-')
        
        processed_events.append(processed_event)
    return processed_events

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb',
    aws_access_key_id=my_secrets.event_agent_aws_access_key_id,
    aws_secret_access_key=my_secrets.event_agent_aws_secret_access_key,
    region_name=my_secrets.event_agent_aws_region
)

# Get DynamoDB tables
event_emails_table = dynamodb.Table('event_emails')
events_table = dynamodb.Table('events')

# Write email data to DynamoDB
def write_to_dynamo(table_name, data):
    """Write data to specified DynamoDB table."""
    try:
        if table_name == "event_emails":
            item = {
                'msg_id': data['msg_id'],
                'received': data['received'],
                'from_email': data['from_email'],
                'subject': data['subject'],
                'body': data['body'],
                'processed': None  # Will be updated when processed
            }
            event_emails_table.put_item(Item=item)
            
        elif table_name == "events":
            # Generate a unique event ID
            event_id = f"{data[0]}_{int(time.time())}"
            
            # Convert date string to timestamp if it's in the correct format
            try:
                event_date = process_datetime(data[1], "timestamp")
                # Check if event date is in the past
                if event_date < int(time.time()):
                    print(f"Skipping past event: {data[0]} on {data[1]}")
                    return
            except:
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
            
            print(f"Writing event to DynamoDB: {item}")  # Debug log
            events_table.put_item(Item=item)
            print(f"Successfully wrote event to DynamoDB")  # Debug log
            
        else:
            print(f"Unknown table: {table_name}")
            
    except Exception as e:
        print(f"Error writing to DynamoDB table {table_name}: {e}")
        if table_name == "events":
            raise

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

# Comprehensive email processing function that handles all email-related operations
def process_emails(action="process_existing", clear_timestamps=False):
    """
    Comprehensive email processing function.
    
    Args:
        action: "fetch_new" to get emails from Gmail, "process_existing" to process stored emails
        clear_timestamps: If True, clears processed timestamps before processing
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
                        processed_count += 1
                    except Exception as e:
                        print(f"Error marking email as read and archived: {e}")

            # Write fetched emails to DynamoDB
            for email in email_data:
                write_to_dynamo("event_emails", email)
            
            return email_data, processed_count
            
        elif action == "process_existing":
            # Clear processed timestamps if requested
            if clear_timestamps:
                try:
                    # Get all emails
                    response = event_emails_table.scan()
                    items = response.get('Items', [])
                    
                    if not items:
                        print("No emails found in the database.")
                        return
                        
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
                            print(f"Error clearing processed timestamp for email {item['msg_id']}: {e}")
                            continue
                    
                    print(f"Cleared processed timestamps from {cleared_count} emails.")
                except Exception as e:
                    print(f"Error in clear_processed_timestamps: {e}")
            
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
                    body_text = email['body']
                    if not body_text:
                        print(f"Skipping email {email['subject']} - no valid text content")
                        continue
                    
                    # Remove HTML tags
                    body_text = re.sub(r'<[^>]+>', ' ', body_text)
                    # Remove email addresses
                    body_text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', body_text)
                    # Remove special characters and extra whitespace
                    body_text = re.sub(r'[^\w\s.,!?-]', ' ', body_text)
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
                            current_time = int(time.time())
                            event_emails_table.update_item(
                                Key={'msg_id': email['msg_id']},
                                UpdateExpression='SET #p = :val',
                                ExpressionAttributeNames={
                                    '#p': 'processed'
                                },
                                ExpressionAttributeValues={
                                    ':val': current_time
                                }
                            )
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

# Run the DynamoDB setup script
def run_setup_script():
    try:
        result = subprocess.run(['python3', 'setup_dynamo.py'], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("DynamoDB tables created successfully!")
        else:
            st.error(f"Error creating tables: {result.stderr}")
    except Exception as e:
        st.error(f"Error running setup script: {e}")

# Get all event emails from DynamoDB
def get_event_emails():
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

# Get all events from DynamoDB
def get_events():
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

# Clear the processed timestamp from all emails in DynamoDB
def clear_processed_timestamps():
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

# Export events from DynamoDB to Google Sheets
def export_to_google_sheets():
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

# Refresh Gmail API credentials by running the OAuth flow again
def refresh_gmail_credentials():
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
            return True
        elif tool == "aws":
            dynamodb.meta.client.list_tables()
            return True
        elif tool == "dynamo":
            existing_tables = dynamodb.meta.client.list_tables()['TableNames']
            return 'event_emails' in existing_tables and 'events' in existing_tables
        elif tool == "google_sheets":
            service = get_service("sheets")
            if service:
                # Try to access the spreadsheet
                service.spreadsheets().get(spreadsheetId=my_secrets.SPREADSHEET_ID).execute()
                return True
            return False
        else:
            print(f"Unknown tool: {tool}")
            return False
    except Exception as e:
        print(f"Error checking {tool} config: {str(e)}")
        return False

def process_datetime(datetime_str, output_format="timestamp"):
    """Unified datetime processing function with multiple output formats."""
    try:
        if output_format == "timestamp":
            # Convert to Unix timestamp
            dt = datetime.strptime(datetime_str, '%m/%d/%Y %I:%M %p')
            return int(dt.timestamp())
        elif output_format == "formatted":
            # Parse email date string and format as MM/DD/YYYY HH:MM AM/PM
            parsed_date = email.utils.parsedate_tz(datetime_str)
            if parsed_date:
                dt = datetime.fromtimestamp(email.utils.mktime_tz(parsed_date))
                return dt.strftime('%m/%d/%Y %I:%M %p')
            return datetime_str
        elif output_format == "date_only":
            # Extract just the date part
            dt = datetime.strptime(datetime_str, '%m/%d/%Y %I:%M %p')
            return dt.strftime('%m/%d/%Y')
        else:
            print(f"Unknown output format: {output_format}")
            return datetime_str
    except Exception as e:
        print(f"Error processing datetime: {str(e)}")
        if output_format == "timestamp":
            return int(time.time())
        return datetime_str

def clear_data(tool):
    """Clear data for specified tool."""
    try:
        if tool == "dynamo":
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
            for collection in collections:
                collection_name = collection.name
                try:
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
                    cleared_count += 1
                except Exception as e:
                    print(f"Failed to clear collection {collection_name}: {str(e)}")
                    continue
            
            if cleared_count > 0:
                print(f"Successfully cleared {cleared_count} collections")
                return True, cleared_count, 0
            else:
                print("No collections were cleared")
                return False, 0, 0
                
        else:
            print(f"Unknown tool: {tool}")
            return False, 0, 0
            
    except Exception as e:
        print(f"Error clearing {tool} data: {str(e)}")
        return False, 0, 0

# Streamlit interface for the Event Agent
def streamlit_interface():
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
            st.metric("Qdrant Status", "⚠️ Not Configured")
    
    with db_action_col:
        # Setup DynamoDB if needed
        if not dynamo_status:
            st.warning("DynamoDB tables are not set up.")
            if st.button("Setup DynamoDB Tables", use_container_width=True):
                with st.spinner("Setting up DynamoDB tables..."):
                    run_setup_script()
                    st.rerun()
        
        # Clear DynamoDB Data
        if st.button("🗑️ Clear DynamoDB Data", use_container_width=True):
            with st.spinner("Clearing DynamoDB data..."):
                success, emails_cleared, events_cleared = clear_data("dynamo")
                if success:
                    st.success(f"DynamoDB cleared successfully! Removed {emails_cleared} emails and {events_cleared} events.")
                    st.rerun()
                else:
                    st.error("Failed to clear DynamoDB data. Check the terminal logs for details.")
        
        # Clear Qdrant Cluster
        if st.button("🗑️ Clear Qdrant Cluster", use_container_width=True):
            with st.spinner("Clearing Qdrant cluster..."):
                if not qdrant_status:
                    st.warning("Qdrant is not configured.")
                else:
                    success, collections_cleared, _ = clear_data("qdrant")
                    if success:
                        st.success(f"Qdrant cluster cleared successfully! Cleared {collections_cleared} collections.")
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
        
        # If Gmail is disconnected, show refresh button
        if not gmail_status:
            if st.button("🔄 Refresh Gmail Credentials", use_container_width=True):
                with st.spinner("Refreshing Gmail credentials..."):
                    if refresh_gmail_credentials():
                        st.success("Credentials refreshed successfully! Please refresh the page.")
                        st.rerun()
                    else:
                        st.error("Failed to refresh credentials. Check the logs for details.")
    
    with gmail_action_col:
        # Check for New Emails
        if st.button("📥 Check for New Emails", use_container_width=True):
            with st.spinner("Fetching new emails from Gmail..."):
                email_data, processed_count = process_emails(action="fetch_new")
                if email_data:
                    for email in email_data:
                        write_to_dynamo("event_emails", email)
                    print(f"Successfully fetched {processed_count} emails from Gmail and written to DynamoDB.")
                else:
                    print("No new emails found in inbox.")
        
        # Process Emails
        if st.button("⚙️ Process Emails", use_container_width=True):
            with st.spinner("Processing emails to extract events..."):
                # First clear processed timestamps
                cleared_count = clear_processed_timestamps()
                if cleared_count > 0:
                    st.info(f"Cleared processed timestamps from {cleared_count} emails.")
                
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
                    write_to_dynamo("event_emails", email)
                print(f"Successfully fetched {processed_count} emails from Gmail and written to DynamoDB.")
            else:
                print("No new emails found in inbox.")
                
            # Then, process emails to extract events
            process_emails()
        except Exception as e:
            print(f"Error in main execution: {e}")
    except Exception as e:
        print(f"Error in Streamlit execution: {e}")
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
import base64
import json
import re
import os
import openai
import sys
from datetime import datetime
import email.utils
import requests
from bs4 import BeautifulSoup
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'secrets')) #REPLACE WITH YOUR PATH
import my_secrets #REPLACE WITH YOUR SECRETS FILE NAME
# print("Available attributes in secrets:", dir(my_secrets)) #HELPS WITH DEBUGGING

# Load credentials for Gmail and Sheets APIs
def load_credentials():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json')
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise Exception("Authorization required. Run Google OAuth process in oauth_setup.py.")
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
    API_KEY = my_secrets.openai_by
    ENDPOINT = "https://api.openai.com/v1/chat/completions"
    
    # Limit text length to approximately 2000 characters to stay within token limits
    # This gives enough context while leaving room for the model's response
    if len(plain_text) > 2000:
        plain_text = plain_text[:2000]
    
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
Only report clean start and end times, not timezones. Report State as a 2 letter abbreviation. Report date as MM/DD/YYYY. The word Location is not a Venue or Address. Return ONLY the JSON array, no other text. If no events are found, return an empty array []."""},
            {"role": "user", "content": plain_text}
        ],
        "temperature": 0.3,
        "max_tokens": 1500  # Increased to handle more events
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print("\nSending text to OpenAI:", plain_text[:200] + "..." if len(plain_text) > 200 else plain_text)
        print("\nUsing API key:", API_KEY[:8] + "..." if API_KEY else "No API key found")
        
        response = requests.post(ENDPOINT, headers=headers, json=payload)
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
            
        print("\nOpenAI raw response:", content)
            
        try:
            # First try direct JSON parsing
            try:
                events = json.loads(content)
                if isinstance(events, list):
                    return clean_events(events)
            except json.JSONDecodeError:
                pass
                
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
                # This regex matches field patterns like '"URL": "value",' and keeps only the last one
                for field in ["URL", "Event Name"]:  # Add other fields if needed
                    pattern = f'"{field}":[^,}}]+,\\s*"{field}":'
                    while re.search(pattern, json_str):
                        match = re.search(pattern, json_str)
                        if match:
                            start, end = match.span()
                            # Keep only the second field
                            second_field_start = json_str.find(f'"{field}":', start + len(field) + 4)
                            json_str = json_str[:start] + json_str[second_field_start:]
                
                # Fix any remaining JSON syntax issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                json_str = re.sub(r'\s+', ' ', json_str)    # Normalize whitespace
                
                print("\nCleaned JSON string:", json_str)
                
                try:
                    events = json.loads(json_str)
                    if not isinstance(events, list):
                        print("Response is not a list")
                        return []
                    return clean_events(events)
                except json.JSONDecodeError as e:
                    print(f"Error parsing cleaned JSON: {e}")
                    print(f"Cleaned JSON string: {json_str}")
                    return []
            else:
                print("No JSON array found in response - start_idx:", start_idx, "end_idx:", end_idx)
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

# Process emails and extract events
def process_emails():
    try:
        creds = load_credentials()
        service = build('sheets', 'v4', credentials=creds)
        
        # Read emails from the Emails sheet
        emails = service.spreadsheets().values().get(
            spreadsheetId=my_secrets.SPREADSHEET_ID,
            range="Emails!A:F"
        ).execute().get('values', [])
        
        if not emails:
            print("No emails found in the Emails sheet.")
            return
        
        # Skip header row and create list of rows to process with their dates
        rows_to_process = []
        for idx, row in enumerate(emails[1:], start=2):
            processed = row[5] if len(row) > 5 else ""  # Column F (index 5) for processed status
            plain_text = row[4] if len(row) > 4 else ""  # Column E (index 4) for body text
            date_str = row[1] if len(row) > 1 else ""  # Column B (index 1) for date
            
            if not processed and plain_text:
                try:
                    # Parse the date string to datetime object for sorting
                    date = datetime.strptime(date_str, '%m/%d/%Y %I:%M %p')
                    rows_to_process.append((idx, plain_text, date))
                except ValueError:
                    print(f"Warning: Could not parse date '{date_str}' for row {idx}")
                    continue
        
        if not rows_to_process:
            print("No unprocessed emails found.")
            return
        
        # Sort rows by date (oldest first)
        rows_to_process.sort(key=lambda x: x[2])
        
        print(f"Processing {len(rows_to_process)} emails in chronological order...")
        
        # Get existing events to check for duplicates
        existing_events = service.spreadsheets().values().get(
            spreadsheetId=my_secrets.SPREADSHEET_ID,
            range="Events!A:J"
        ).execute().get('values', [])
        
        # Create URL to row mapping
        url_to_row = {}
        if existing_events:
            for row in existing_events[1:]:  # Skip header
                if len(row) > 9:  # URL is in column J (index 9)
                    url = sanitize_url(row[9])
                    if url:
                        url_to_row[url] = row
        
        for row_num, text, _ in rows_to_process:  # Ignore the date in the loop
            try:
                events = extract_events_with_ai(text)
                if events:
                    for event in events:
                        try:
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
                            
                            url = event_data[9]  # Get the URL
                            if not url:
                                continue
                                
                            # Check if URL already exists
                            if url in url_to_row:
                                # Update existing event with any missing data
                                update_existing_event(service, my_secrets.SPREADSHEET_ID, event_data, url_to_row[url])
                            else:
                                # Identify missing fields
                                missing_fields = []
                                field_names = ["Event Name", "Date", "Start Time", "End Time", "City", "State", "Venue", "Address", "Description"]
                                for i, field in enumerate(event_data[:-1]):  # Exclude URL
                                    if not field:
                                        missing_fields.append(field_names[i])
                                
                                # If there are missing fields, fetch data from URL
                                if missing_fields:
                                    print(f"Fetching missing data for URL: {url}")
                                    print(f"Missing fields: {', '.join(missing_fields)}")
                                    missing_data = fetch_missing_data(url, missing_fields)
                                    if missing_data:
                                        # Update missing fields
                                        for field_name, value in missing_data.items():
                                            field_index = field_names.index(field_name)
                                            if not event_data[field_index]:
                                                event_data[field_index] = value
                                
                                # Append new event
                                service.spreadsheets().values().append(
                                    spreadsheetId=my_secrets.SPREADSHEET_ID,
                                    range="Events!A:J",
                                    valueInputOption="RAW",
                                    insertDataOption="INSERT_ROWS",
                                    body={"values": [event_data]}
                                ).execute()
                                
                                # Update URL mapping
                                url_to_row[url] = event_data
                        except Exception as e:
                            print(f"Error processing event: {e}")
                            continue
                    
                    # Mark email as processed with timestamp
                    timestamp = datetime.utcnow().isoformat()
                    service.spreadsheets().values().update(
                        spreadsheetId=my_secrets.SPREADSHEET_ID,
                        range=f"Emails!F{row_num}",
                        valueInputOption="RAW",
                        body={"values": [[timestamp]]}
                    ).execute()
            except Exception as e:
                print(f"Error processing email row {row_num}: {e}")
                continue
        
        # Sort events by date and time
        data_range = "Events!A:J"
        sheet_data = service.spreadsheets().values().get(
            spreadsheetId=my_secrets.SPREADSHEET_ID,
            range=data_range
        ).execute().get('values', [])
        
        if sheet_data:
            headers = sheet_data[0]
            rows = sheet_data[1:]
            rows.sort(key=lambda x: (x[1], x[2], x[0]))  # Sort by Date, Start Time, Event Name
            
            service.spreadsheets().values().update(
                spreadsheetId=my_secrets.SPREADSHEET_ID,
                range=data_range,
                valueInputOption="RAW",
                body={"values": [headers] + rows}
            ).execute()
        
    except Exception as e:
        print(f"Error in process_emails: {e}")
        raise

# Main function
if __name__ == "__main__":
    try:
        # First, get emails from Gmail
        email_data, processed_count = extract_emails()
        if email_data:
            write_to_sheet(email_data)
            print(f"Successfully fetched {processed_count} emails from Gmail and written to Emails sheet.")
        else:
            print("No new emails found in inbox.")
            
        # Then, process emails to extract events
        process_emails()
    except Exception as e:
        print(f"Error: {e}")
import boto3
import sys
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Add the secrets directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../secrets')))
import my_secrets

# Gmail API SCOPES
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/spreadsheets'
]

def setup_oauth():
    """Setup Google OAuth credentials."""
    print("Setting up Google OAuth credentials...")
    
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            # Verify the credentials have all required scopes
            if not all(scope in creds.scopes for scope in SCOPES):
                print("Missing required scopes, refreshing credentials...")
                creds = None
        except Exception as e:
            print(f"Error loading existing credentials: {e}")
            creds = None
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                print("Refreshed existing credentials.")
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                creds = None
        
        if not creds or not creds.valid:
            if not os.path.exists('credentials.json'):
                print("ERROR: credentials.json not found!")
                print("Please download your Google API credentials file and save it as 'credentials.json' in this directory.")
                return False
            
            try:
                print("Starting OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                print("OAuth flow completed successfully.")
            except Exception as e:
                print(f"Error in OAuth flow: {e}")
                return False
        
        # Save the credentials for the next run
        try:
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            print("Credentials saved successfully to token.json")
        except Exception as e:
            print(f"Error saving credentials: {e}")
            return False
    
    # Test the credentials
    try:
        service = build('gmail', 'v1', credentials=creds)
        service.users().labels().list(userId='me').execute()
        print("✅ OAuth credentials are valid and working!")
        return True
    except Exception as e:
        print(f"❌ Error testing OAuth credentials: {e}")
        return False

def setup_dynamodb():
    """Setup DynamoDB tables."""
    print("Setting up DynamoDB tables...")
    
    try:
        # Initialize DynamoDB client
        dynamodb = boto3.resource('dynamodb',
            aws_access_key_id=my_secrets.event_agent_aws_access_key_id,
            aws_secret_access_key=my_secrets.event_agent_aws_secret_access_key,
            region_name=my_secrets.event_agent_aws_region
        )
        
        # Test connection
        dynamodb.meta.client.list_tables()
        print("✅ DynamoDB connection successful!")
        
    except Exception as e:
        print(f"❌ Error connecting to DynamoDB: {e}")
        print("Please check your AWS credentials in my_secrets.py")
        return False
    
    def create_table(table_name, key_schema, attribute_definitions):
        """Create a DynamoDB table if it doesn't exist."""
        try:
            # Check if table exists
            existing_tables = dynamodb.meta.client.list_tables()['TableNames']
            if table_name in existing_tables:
                print(f"✅ Table {table_name} already exists.")
                return True
            
            # Create table
            print(f"Creating table {table_name}...")
            table = dynamodb.create_table(
                TableName=table_name,
                KeySchema=key_schema,
                AttributeDefinitions=attribute_definitions,
                BillingMode='PAY_PER_REQUEST'
            )
            
            # Wait for table to be created
            table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
            print(f"✅ Table {table_name} created successfully.")
            return True
            
        except Exception as e:
            print(f"❌ Error creating table {table_name}: {e}")
            return False
    
    # Create events table only (email data will be stored in Qdrant)
    events_key_schema = [
        {'AttributeName': 'event_id', 'KeyType': 'HASH'}  # Partition key
    ]
    events_attributes = [
        {'AttributeName': 'event_id', 'AttributeType': 'S'}
    ]
    events_success = create_table('events', events_key_schema, events_attributes)
    
    return events_success

def setup_google_sheets():
    """Test Google Sheets access."""
    print("Testing Google Sheets access...")
    
    try:
        # Load credentials
        if not os.path.exists('token.json'):
            print("❌ No OAuth credentials found. Please run OAuth setup first.")
            return False
        
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds.valid:
            print("❌ OAuth credentials are invalid. Please run OAuth setup first.")
            return False
        
        # Test Google Sheets access
        service = build('sheets', 'v4', credentials=creds)
        service.spreadsheets().get(spreadsheetId=my_secrets.SPREADSHEET_ID).execute()
        print("✅ Google Sheets access successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error accessing Google Sheets: {e}")
        print("Please check your SPREADSHEET_ID in my_secrets.py")
        return False

def setup_qdrant():
    """Setup Qdrant for email storage and vector search."""
    print("Setting up Qdrant for email storage...")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        
        if not hasattr(my_secrets, 'QDRANT_URL') or not hasattr(my_secrets, 'QDRANT_API_KEY'):
            print("❌ Qdrant credentials not configured in my_secrets.py")
            print("Please add QDRANT_URL and QDRANT_API_KEY to your my_secrets.py file")
            return False
        
        client = QdrantClient(url=my_secrets.QDRANT_URL, api_key=my_secrets.QDRANT_API_KEY)
        
        # Test connection
        client.get_collections()
        print("✅ Qdrant connection successful!")
        
        # Create email collection if it doesn't exist
        collection_name = "emails"
        try:
            collections = client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)
            
            if not collection_exists:
                print(f"Creating collection '{collection_name}'...")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI embedding size
                        distance=models.Distance.COSINE
                    )
                )
                print(f"✅ Collection '{collection_name}' created successfully!")
            else:
                print(f"✅ Collection '{collection_name}' already exists.")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating Qdrant collection: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return False

def main():
    """Main setup function that runs all setup tasks."""
    print("🚀 Event Agent Setup")
    print("=" * 50)
    
    # Check if secrets file exists
    if not os.path.exists(os.path.join('..', 'secrets', 'my_secrets.py')):
        print("❌ my_secrets.py not found in ../secrets/ directory!")
        print("Please create your secrets file first.")
        return
    
    # Run all setup tasks
    oauth_success = setup_oauth()
    dynamo_success = setup_dynamodb()
    sheets_success = setup_google_sheets()
    qdrant_success = setup_qdrant()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"OAuth Credentials: {'✅' if oauth_success else '❌'}")
    print(f"DynamoDB Tables: {'✅' if dynamo_success else '❌'}")
    print(f"Google Sheets: {'✅' if sheets_success else '❌'}")
    print(f"Qdrant Email Storage: {'✅' if qdrant_success else '❌'}")
    
    if oauth_success and dynamo_success and sheets_success and qdrant_success:
        print("\n🎉 Setup completed successfully! You can now run the Event Agent.")
    else:
        print("\n⚠️ Some setup tasks failed. Please check the errors above and try again.")
        if not oauth_success:
            print("   - Make sure you have credentials.json in this directory")
        if not dynamo_success:
            print("   - Check your AWS credentials in my_secrets.py")
        if not sheets_success:
            print("   - Check your SPREADSHEET_ID in my_secrets.py")
        if not qdrant_success:
            print("   - Check your QDRANT_URL and QDRANT_API_KEY in my_secrets.py")

if __name__ == "__main__":
    main() 
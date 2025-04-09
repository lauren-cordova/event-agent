# 🤖 Event Agent

An intelligent agent that processes event-related emails, extracts event details, and stores them in DynamoDB. Built with Python, Gmail API, OpenAI, and AWS.

## Features

- 📧 **Email Processing**: Automatically fetches and processes event-related emails from Gmail
- 🤖 **AI-Powered Extraction**: Uses OpenAI to intelligently extract event details from email content
- 🗄️ **DynamoDB Storage**: Stores emails and events in AWS DynamoDB for efficient querying
- 🎯 **Event Management**: Tracks and manages events with detailed information
- 🎨 **Streamlit Interface**: Beautiful web interface for monitoring and managing the system

## Architecture

```mermaid
graph TD
    A[Gmail Inbox] -->|Fetch Emails| B[Email Parser]
    B -->|Extract Content| C[Event Extractor]
    C -->|AI Processing| D[Data Enricher]
    D -->|Store Data| E[DynamoDB]
    E -->|Query Data| F[Streamlit UI]
    F -->|Display| G[User Interface]
    
    style A fill:#EA4335,color:#fff
    style B fill:#4285F4,color:#fff
    style C fill:#FBBC05,color:#000
    style D fill:#34A853,color:#fff
    style E fill:#FF9900,color:#fff
    style F fill:#00A67E,color:#fff
    style G fill:#1DA1F2,color:#fff
```

## Prerequisites

- Python 3.8+
- AWS Account with DynamoDB access
- Gmail Account
- OpenAI API Key

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/event-agent.git
   cd event-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up credentials**
   - Copy `my_secrets_template.py` to `my_secrets.py`:
     ```bash
     cp my_secrets_template.py my_secrets.py
     ```
   - Fill in your credentials in `my_secrets.py`:
     ```python
     # Gmail API credentials
     CLIENT_ID = "your_client_id"
     CLIENT_SECRET = "your_client_secret"
     
     # AWS credentials
     aws_access_key_id = "your_aws_access_key"
     aws_secret_access_key = "your_aws_secret_key"
     aws_region = "your_aws_region"
     
     # OpenAI API key
     openai_by = "your_openai_api_key"
     ```
   - ⚠️ **Important**: Never commit `my_secrets.py` to version control. It's already in `.gitignore`.

4. **Set up Gmail API**
   - Enable Gmail API in Google Cloud Console
   - Download credentials.json
   - Run OAuth setup:
     ```bash
     python3 oauth_setup.py
     ```

5. **Set up DynamoDB**
   - Run the setup script:
     ```bash
     python3 setup_dynamo.py
     ```
   - This creates two tables:
     - `event_emails`: Stores email data
     - `events`: Stores extracted event information

## Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run main.py
   ```

2. **Using the Interface**
   - Check system status (Gmail, AWS, DynamoDB)
   - Setup DynamoDB tables if needed
   - Process new emails
   - View event emails and extracted events
   - Filter and sort events

## Data Storage

### Event Emails Table
- `msg_id`: Unique message ID (Primary Key)
- `received`: Timestamp when email was received
- `sender`: Email sender
- `subject`: Email subject
- `body`: Email body content
- `processed`: Timestamp when email was processed

### Events Table
- `event_id`: Unique event ID (Primary Key)
- `event_name`: Name of the event
- `date`: Event date
- `start_time`: Event start time
- `end_time`: Event end time
- `city`: Event city
- `state`: Event state
- `venue`: Event venue
- `address`: Event address
- `description`: Event description
- `url`: Event URL

## Troubleshooting

### Gmail API Issues
1. **Token Expired/Revoked**
   - Delete `token.json`
   - Run `python3 oauth_setup.py`
   - Re-authenticate with Gmail

2. **Insufficient Permissions**
   - Ensure you have granted all required scopes:
     - `https://www.googleapis.com/auth/gmail.readonly`
     - `https://www.googleapis.com/auth/gmail.modify`

### DynamoDB Issues
1. **Tables Not Found**
   - Run `python3 setup_dynamo.py`
   - Verify AWS credentials in `my_secrets.py`

2. **Access Denied**
   - Check AWS IAM permissions
   - Verify region matches your DynamoDB tables

## Security

- Never commit `my_secrets.py` or `token.json`
- Use `.gitignore` to exclude sensitive files
- Rotate AWS and Gmail credentials regularly
- Keep dependencies updated

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

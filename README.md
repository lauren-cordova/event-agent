# 🎫 Event Agent

An intelligent event agent that processes emails from Gmail, extracts event URLs, and populates event data into Google Sheets. Built with Python, Gmail API, OpenAI, and Google Sheets API. The application provides a beautiful Streamlit interface for managing the entire workflow.

## Features

- 📧 **Email Processing**: Automatically fetches and processes emails from Gmail inbox
- 🔗 **URL Extraction**: Extracts URLs from email content (both HTML links and text)
- 🧹 **URL Cleaning**: Removes URL parameters to ensure uniqueness
- 🎯 **Event Data Population**: Uses OpenAI to intelligently extract event information from web pages
- 📊 **Google Sheets Integration**: Directly populates event data into Google Sheets
- 🎨 **Beautiful UI**: Modern Streamlit interface with status monitoring
- 🔄 **Connection Management**: Easy Gmail connection refresh functionality

## Event Data Fields

The application extracts and populates the following event information:

- **Event Name** (Column A): Name or title of the event
- **Date** (Column B): Event date in mm/dd/yyyy format
- **Start Time** (Column C): Start time in hh:mm AM/PM format
- **End Time** (Column D): End time in hh:mm AM/PM format
- **City** (Column E): City where the event takes place
- **State** (Column F): 2-letter state abbreviation
- **Venue** (Column G): Venue name
- **Address** (Column H): Full address of the event
- **Description** (Column I): Brief description of the event
- **URL** (Column J): Source URL from email

## Prerequisites

- Python 3.8+
- Gmail Account
- OpenAI API Key
- Google Cloud project with:
  - Gmail API enabled
  - Google Sheets API enabled
  - OAuth 2.0 credentials configured

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
     
     # Google Sheets
     SPREADSHEET_ID = "your_spreadsheet_id_here"
     
     # OpenAI API key
     openai_by = "your_openai_api_key"
     ```
   - ⚠️ **Important**: Never commit `my_secrets.py` to version control. It's already in `.gitignore`.

4. **Set up Gmail API**
   - Enable Gmail API in Google Cloud Console
   - Download credentials.json and place it in the project root
   - The first time you run the app, it will prompt for OAuth authentication

5. **Create Google Sheet**
   - Create a new Google Sheet
   - Add a sheet named "Events"
   - Add headers in row 1: Event Name, Date, Start Time, End Time, City, State, Venue, Address, Description, URL
   - Copy the spreadsheet ID from the URL and add it to `my_secrets.py`

## Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run main.py
   ```

2. **Using the Interface**
   - **Process Emails**: Fetches emails from Gmail and extracts unique URLs to the Google Sheet
   - **Scrape Event Data**: Goes through URLs in the sheet and populates event information using AI
   - **Refresh Gmail Connection**: Refreshes OAuth tokens if needed
   - **Quick Links**: Direct links to Gmail and Google Sheets
   - **System Status**: Shows configuration status

## Workflow

1. **Email Processing**: 
   - Connects to Gmail API
   - Fetches recent emails from inbox
   - Extracts URLs from HTML and text content
   - Removes URL parameters for uniqueness
   - Adds new unique URLs to Google Sheet

2. **Event Data Population**:
   - Reads URLs from Google Sheet
   - Scrapes each webpage
   - Uses OpenAI to extract structured event data
   - Populates all event fields in the sheet

## Troubleshooting

### Gmail API Issues
- **Token Expired**: Use the "Refresh Gmail Connection" button
- **Credentials Missing**: Ensure `credentials.json` is in the project root
- **API Not Enabled**: Enable Gmail API in Google Cloud Console

### Google Sheets Issues
- **Permission Denied**: Ensure the service account has edit access to the sheet
- **Sheet Not Found**: Verify the spreadsheet ID in `my_secrets.py`
- **Wrong Sheet Name**: Ensure the sheet is named "Events"

### OpenAI Issues
- **API Key Invalid**: Check your OpenAI API key in `my_secrets.py`
- **Rate Limits**: The app processes URLs sequentially to avoid rate limits

## File Structure

```
event-agent/
├── main.py                 # Main Streamlit application
├── my_secrets_template.py  # Template for secrets configuration
├── my_secrets.py          # Your actual secrets (not in git)
├── credentials.json       # Gmail API credentials (not in git)
├── token.pickle          # OAuth tokens (not in git)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

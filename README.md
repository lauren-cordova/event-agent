# 🤖 Event Agent

Working on some scripts to build a personal agent for social or professional event tracking and management. This is very much a work in progress, so please check back for updates and feel free to leave comments, feedback, or requests! ✨

## 🚀 Quick Start

1. Clone this repository:
```bash
git clone https://github.com/yourusername/event-agent.git
cd event-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your credentials:
   - Copy `my_secrets_template.py` to `my_secrets.py`
   - Fill in your API keys and credentials in `my_secrets.py`
   - Never commit `my_secrets.py` to version control
   - Keep your credentials secure and private

4. Set up Google OAuth:
   - Go to the [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select an existing one
   - Enable the Gmail and Google Sheets APIs
   - Create OAuth 2.0 credentials
   - Download the credentials and save as `credentials.json`
   - Run the OAuth setup script:
     ```bash
     python3 oauth_setup.py
     ```
   - Follow the authentication flow in your browser

## 🛠️ Agent Functionality

### 🎉 Event Tracking & Management
The **Event Agent** is designed to help you track and manage your social and professional events. 

**🌟 Key Features:**
- **Email Integration:** Get notified about upcoming events so you never miss out! ⏰
- **Event Categorization:** Organize events by type (e.g., work, social) for easy access. 📅
- **Integration with APIs:** Future updates will include support for various APIs to enhance functionality. 🔗

## 🔧 Troubleshooting

### 🔑 OAuth Token Issues
If you encounter errors like "Token has been expired or revoked" or other authentication issues, follow these steps to refresh your OAuth token:

1. Delete the existing token file:
```bash
rm -f token.json
```

2. Run the OAuth setup script:
```bash
python3 oauth_setup.py
```

3. Follow the authentication flow:
   - Click the URL that appears in the terminal
   - Sign in with your Google account
   - Grant the requested permissions
   - Wait for the "Authentication successful" message

After completing these steps, your token will be renewed and the script should work again.

## 🔒 Security Best Practices
- Never commit your credentials or API keys to version control
- Keep your `my_secrets.py` file secure and private
- Regularly rotate your API keys and tokens
- Use environment variables for sensitive data in production
- Review the `.gitignore` file to ensure sensitive files are not tracked

## 💡 Stay Tuned for More Updates!
Feel free to open an issue or submit a pull request with feedback or feature suggestions. 🌟
import requests
import os
from dotenv import load_dotenv

load_dotenv()


def send_pushover_notification(title, message, priority=0):
    pushover_user_key = os.getenv('PUSHOVER_USER_KEY')
    pushover_api_token = os.getenv('PUSHOVER_API_TOKEN')
    
    if not pushover_user_key or not pushover_api_token:
        print("Warning: Pushover credentials not found in environment variables")
        return

    payload = {
        "token": pushover_api_token,
        "user": pushover_user_key,
        "message": message,
        "title": title,
        "priority": priority,
    }

    response = requests.post("https://api.pushover.net/1/messages.json", data=payload)

    if response.status_code != 200:
        print("Pushover notification failed:", response.text)

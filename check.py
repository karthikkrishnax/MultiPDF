import os
import dotenv

print(os.getenv("OPENAI_API_KEY")) 

from dotenv import load_dotenv
import os

# Specify the path to your .env file if it's in the same directory
load_dotenv(dotenv_path=".env")

# Test to see if the key loads
print(os.getenv("OPENAI_API_KEY"))
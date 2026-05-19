import os
from dotenv import load_dotenv
load_dotenv()

print("ENDPOINT:", os.getenv("PREDICTION_ENDPOINT"))
print("KEY:", os.getenv("PREDICTION_KEY"))
print("PROJECT_ID:", os.getenv("PREDICTION_PROJECT_ID"))
print("ITERATION:", os.getenv("PREDICTION_ITERATION_NAME"))
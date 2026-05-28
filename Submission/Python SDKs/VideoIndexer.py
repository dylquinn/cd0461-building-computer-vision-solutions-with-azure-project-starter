import os
import io
import time
import glob
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

from video_indexer import VideoIndexer
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceClient, FaceAdministrationClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
)

# ── Load Environment Variables ────────────────────────────────────────────────
load_dotenv()

FACE_KEY              = os.getenv("FACE_KEY")
FACE_ENDPOINT         = os.getenv("FACE_ENDPOINT")
VI_SUBSCRIPTION_KEY   = os.getenv("VI_SUBSCRIPTION_KEY")
VI_LOCATION           = os.getenv("VI_LOCATION")
VI_ACCOUNT_ID         = os.getenv("VI_ACCOUNT_ID")

# ── Clients ───────────────────────────────────────────────────────────────────
face_client       = FaceClient(endpoint=FACE_ENDPOINT, credential=AzureKeyCredential(FACE_KEY))
face_admin_client = FaceAdministrationClient(endpoint=FACE_ENDPOINT, credential=AzureKeyCredential(FACE_KEY))

# ── Video Indexer Setup ───────────────────────────────────────────────────────
video_analysis = VideoIndexer(
    vi_subscription_key="",
    vi_location="trial",
    vi_account_id=VI_ACCOUNT_ID
)

# Authenticate with access token instead
video_analysis.check_access_token()

# ── TASK 1: Upload Video ──────────────────────────────────────────────────────
# Using the sample video from the Udacity starter repo
VIDEO_URL = "https://raw.githubusercontent.com/dylquinn/cd0461-building-computer-vision-solutions-with-azure-project-starter/master/starter/digital-video-sample/avkash-boarding-pass.mp4"

uploaded_video_id = video_analysis.upload_to_video_indexer(
    input_filename=VIDEO_URL,
    video_name='avkash-boarding-pass',
    video_language='English'
)
print("Uploaded video ID:", uploaded_video_id)

# ── Wait for processing ───────────────────────────────────────────────────────
print("Waiting for video to process...")
time.sleep(60)  # increase if needed — 60s is usually enough for a 30s video

# ── Get video info ────────────────────────────────────────────────────────────
info = video_analysis.get_video_info(uploaded_video_id, video_language='English')

# ── TASK 2: Extract Face Thumbnails ──────────────────────────────────────────
faces_info = info['videos'][0]['insights']['faces'][0]['thumbnails']
print("We found {} faces in this video.".format(len(faces_info)))

images = []

for each_thumb in faces_info:
    if 'fileName' in each_thumb and 'id' in each_thumb:
        thumb_id = each_thumb['id']
        img_code = video_analysis.get_thumbnail_from_video_indexer(uploaded_video_id, thumb_id)
        img      = Image.open(io.BytesIO(img_code))
        images.append(img)

# Display and save thumbnails locally
for i, img in enumerate(images, start=1):
    filename = f'human-face{i}.jpg'
    img.save(filename)
    print(f"Saved: {filename}")
    plt.figure()
    plt.imshow(img)
    plt.show()

# ── TASK 4: Emotions and Sentiments ──────────────────────────────────────────
print("\nSentiments:", info['summarizedInsights']['sentiments'])
print("Emotions:",   info['summarizedInsights']['emotions'])

# ── Summary for Script 2 and validation script ───────────────────────────────
saved_faces = [f for f in glob.glob('*.jpg') if f.startswith("human-face")]
print("\nFace images saved — use human-face1.jpg in validation_and_kiosk_final.py:")
for f in saved_faces:
    print(" ", f)
print("\nVideo ID:", uploaded_video_id)
import io
import time
import glob
import matplotlib.pyplot as plt
from PIL import Image

from video_indexer import VideoIndexer
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceClient, FaceAdministrationClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
)

# ── Credentials ───────────────────────────────────────────────────────────────
FACE_KEY      = "ENTER FACE SERVICE RESOURCE KEY"
FACE_ENDPOINT = "ENTER FACE SERVICE RESOURCE ENDPOINT"

face_client       = FaceClient(endpoint=FACE_ENDPOINT, credential=AzureKeyCredential(FACE_KEY))
face_admin_client = FaceAdministrationClient(endpoint=FACE_ENDPOINT, credential=AzureKeyCredential(FACE_KEY))

CONFIG = {
    'SUBSCRIPTION_KEY': 'YOUR VIDEO INDEXER SUBSCRIPTION KEY',
    'LOCATION': 'trial',
    'ACCOUNT_ID': 'YOUR VIDEO INDEXER ACCOUNT ID'
}

# ── Video Indexer Setup ───────────────────────────────────────────────────────
video_analysis = VideoIndexer(
    vi_subscription_key=CONFIG['SUBSCRIPTION_KEY'],
    vi_location=CONFIG['LOCATION'],
    vi_account_id=CONFIG['ACCOUNT_ID']
)

# ── TASK 1: Upload Video ──────────────────────────────────────────────────────
uploaded_video_id = video_analysis.upload_to_video_indexer(
    input_filename='/ENTER/YOUR/VIDEO/FILE/PATH/HERE.mp4',
    video_name='DYLAN-11-second',
    video_language='English'
)
print("Uploaded video ID:", uploaded_video_id)

# ── Wait for processing ───────────────────────────────────────────────────────
print("Waiting for video to process...")
time.sleep(60)  # increase if video is longer than ~30 seconds

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

# Display and save thumbnails
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

# ── Print saved filenames for use in Script 2 ─────────────────────────────────
saved_faces = [f for f in glob.glob('*.jpg') if f.startswith("human-face")]
print("\nFace images saved — use these in Script 2:")
for f in saved_faces:
    print(" ", f)
print("\nVideo ID to carry over to Script 2:", uploaded_video_id)
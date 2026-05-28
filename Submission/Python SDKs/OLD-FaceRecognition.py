import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file automatically

FORM_RECOGNIZER_ENDPOINT  = os.getenv("FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY       = os.getenv("FORM_RECOGNIZER_KEY")
FACE_ENDPOINT             = os.getenv("FACE_ENDPOINT")
FACE_KEY                  = os.getenv("FACE_KEY")
PREDICTION_ENDPOINT       = os.getenv("PREDICTION_ENDPOINT")
PREDICTION_KEY            = os.getenv("PREDICTION_KEY")
PREDICTION_PROJECT_ID     = os.getenv("PREDICTION_PROJECT_ID")
PREDICTION_ITERATION_NAME = os.getenv("PREDICTION_ITERATION_NAME")
TRAINING_DATA_SAS_URL     = os.getenv("TRAINING_DATA_SAS_URL")
BOARDING_PASS_MODEL_ID    = os.getenv("BOARDING_PASS_MODEL_ID")

import datetime
import pandas as pd
from PIL import Image, ImageDraw
import requests
import io
import glob, os, sys, time, uuid

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from urllib.parse import urlparse
from io import BytesIO

from video_indexer import VideoIndexer
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceClient, FaceAdministrationClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeTypeDetection03,
    FaceAttributeTypeRecognition04,
    QualityForRecognition,
)

# ── Credentials ──────────────────────────────────────────────────────────────
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

# Upload video and get info
uploaded_video_id = video_analysis.upload_to_video_indexer(
    input_filename='/ENTER/YOUR/VIDEO/FILE/PATH/HERE.mp4',
    video_name='DYLAN-11-second',
    video_language='English'
)

print("Waiting for video to process...")
time.sleep(60)  # adjust depending on video length

info = video_analysis.get_video_info(uploaded_video_id, video_language='English')

# ── Extract Face Thumbnails from Video ───────────────────────────────────────
video_id = uploaded_video_id  # reuse from upload, or hardcode if resuming a session

info = video_analysis.get_video_info(video_id, video_language='English')

faces_info = info['videos'][0]['insights']['faces'][0]['thumbnails']
print("We found {} faces in this video.".format(len(faces_info)))

images    = []
img_strs  = []

for each_thumb in faces_info:
    if 'fileName' in each_thumb and 'id' in each_thumb:
        thumb_id  = each_thumb['id']
        img_code  = video_analysis.get_thumbnail_from_video_indexer(video_id, thumb_id)
        img_strs.append(img_code)
        img       = Image.open(io.BytesIO(img_code))
        images.append(img)

# Display and save thumbnails
for i, img in enumerate(images, start=1):
    plt.figure()
    plt.imshow(img)
    plt.show()
    img.save(f'human-face{i}.jpg')

# ── Video Insights ────────────────────────────────────────────────────────────
print("Sentiments:", info['summarizedInsights']['sentiments'])
print("Emotions:",   info['summarizedInsights']['emotions'])

# ── Build Large Person Group (updated from PersonGroup) ───────────────────────
person_group_id   = str(uuid.uuid4())
person_group_name = 'ENTER_GROUP_NAME_HERE'

def build_person_group(admin_client, person_group_id, group_name):
    print('Creating person group:', person_group_id)

    # Create Large Person Group (replaces PersonGroup in new SDK)
    admin_client.large_person_group.create(
        large_person_group_id=person_group_id,
        name=group_name
    )

    # Create a person inside the group
    human_person = admin_client.large_person_group.create_person(
        large_person_group_id=person_group_id,
        name=group_name
    )

    # Add face images to the person
    human_face_images = [f for f in glob.glob('*.jpg') if f.startswith("human-face")]
    for image_path in human_face_images:
        with open(image_path, 'rb') as img_file:
            admin_client.large_person_group.add_face(
                large_person_group_id=person_group_id,
                person_id=human_person.person_id,
                image_content=img_file,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04
            )

    # Train the group
    poller = admin_client.large_person_group.begin_train(person_group_id)
    poller.result()
    print('Person group trained successfully.')

build_person_group(face_admin_client, person_group_id, person_group_name)

# ── Detect Faces in Query Images ──────────────────────────────────────────────
def detect_faces(client, query_images_list):
    print('Detecting faces in query images...')
    face_ids = {}

    for image_name in query_images_list:
        with open(image_name, 'rb') as img_file:
            time.sleep(5)  # avoid rate limiting
            detected_faces = client.detect(
                image_content=img_file,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=True
            )

        for face in detected_faces:
            print(f'Face ID {face.face_id} found in {image_name}')
            face_ids[image_name] = face.face_id  # assumes one face per image

    return face_ids

test_images = [f for f in glob.glob('*.jpg') if f.startswith("human-face")]
ids = detect_faces(face_client, test_images)

# ── Verify Two Faces Match ────────────────────────────────────────────────────
verify_result = face_client.verify_face_to_face(
    face_id1=ids['human-face1.jpg'],
    face_id2=ids['human-face2.jpg']
)

if verify_result.is_identical:
    print(f"Same person ✓ — confidence: {verify_result.confidence}")
else:
    print(f"Different persons ✗ — confidence: {verify_result.confidence}")

# ── Detect Face from Driving License URL ──────────────────────────────────────
def show_image_in_cell(face_url):
    response = requests.get(face_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()

dl_source_url = 'ENTER_DL_IMAGE_URL_HERE'
show_image_in_cell(dl_source_url)

dl_faces = face_client.detect(
    url=dl_source_url,
    detection_model=FaceDetectionModel.DETECTION03,
    recognition_model=FaceRecognitionModel.RECOGNITION04,
    return_face_id=True
)

for face in dl_faces:
    print(f'Face ID {face.face_id} found in DL image')
    ids['ca-dl-sample.png'] = face.face_id

# ── Verify DL Face Against Video Thumbnail ────────────────────────────────────
dl_verify_result = face_client.verify_face_to_face(
    face_id1=ids['human-face1.jpg'],
    face_id2=ids['ca-dl-sample.png']
)

if dl_verify_result.is_identical:
    print(f"DL matches video face ✓ — confidence: {dl_verify_result.confidence}")
else:
    print(f"DL does not match video face ✗ — confidence: {dl_verify_result.confidence}")

# ── Draw Face Rectangles ──────────────────────────────────────────────────────
def get_rectangle(face):
    rect   = face.face_rectangle
    left   = rect.left
    top    = rect.top
    right  = left + rect.width
    bottom = top  + rect.height
    return ((left, top), (right, bottom))

def draw_face_rectangles(source_url, detected_faces):
    response = requests.get(source_url)
    img      = Image.open(BytesIO(response.content))
    draw     = ImageDraw.Draw(img)
    for face in detected_faces:
        draw.rectangle(get_rectangle(face), outline='red', width=10)
    return img

annotated = draw_face_rectangles(dl_source_url, dl_faces)
plt.figure(figsize=(10, 5))
plt.imshow(annotated)
plt.show()

# ── Identify Face Against Person Group ───────────────────────────────────────
dl_face_id = ids['ca-dl-sample.png']

identify_results = face_client.identify_from_large_person_group(
    face_ids=[dl_face_id],
    large_person_group_id=person_group_id
)

for result in identify_results:
    for candidate in result.candidates:
        print(f"Identity match confidence: {candidate.confidence}")
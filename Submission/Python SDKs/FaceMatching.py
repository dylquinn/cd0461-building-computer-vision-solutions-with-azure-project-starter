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

import glob
import time
import uuid
import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from io import BytesIO

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

# ── TASK 3: Build Person Group from Video Thumbnails ──────────────────────────
person_group_id   = str(uuid.uuid4())
person_group_name = 'ENTER_GROUP_NAME_HERE'

def build_person_group(admin_client, person_group_id, group_name):
    print('Creating person group:', person_group_id)

    admin_client.large_person_group.create(
        large_person_group_id=person_group_id,
        name=group_name
    )

    human_person = admin_client.large_person_group.create_person(
        large_person_group_id=person_group_id,
        name=group_name
    )

    human_face_images = [f for f in glob.glob('*.jpg') if f.startswith("human-face")]
    print(f"Adding {len(human_face_images)} face images to person group...")

    for image_path in human_face_images:
        with open(image_path, 'rb') as img_file:
            admin_client.large_person_group.add_face(
                large_person_group_id=person_group_id,
                person_id=human_person.person_id,
                image_content=img_file,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04
            )
        print(f"  Added: {image_path}")

    poller = admin_client.large_person_group.begin_train(person_group_id)
    poller.result()
    print('Person group trained successfully.')
    return human_person

human_person = build_person_group(face_admin_client, person_group_id, person_group_name)

# ── Detect Faces in Saved Thumbnails ─────────────────────────────────────────
def detect_faces(client, query_images_list):
    print('\nDetecting faces in query images...')
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
            face_ids[image_name] = face.face_id

    return face_ids

test_images = [f for f in glob.glob('*.jpg') if f.startswith("human-face")]
ids = detect_faces(face_client, test_images)

# Print detected face IDs so you can confirm before verification
print("\nDetected face IDs:")
for name, fid in ids.items():
    print(f"  {name}: {fid}")

# ── Verify Two Video Faces Match Each Other ───────────────────────────────────
# Update these filenames based on what Script 1 printed
face_image_1 = 'human-face1.jpg'
face_image_2 = 'human-face2.jpg'

if face_image_1 in ids and face_image_2 in ids:
    verify_result = face_client.verify_face_to_face(
        face_id1=ids[face_image_1],
        face_id2=ids[face_image_2]
    )
    if verify_result.is_identical:
        print(f"\nSame person ✓ — confidence: {verify_result.confidence}")
    else:
        print(f"\nDifferent persons ✗ — confidence: {verify_result.confidence}")
else:
    print("Could not find both face images for verification — check filenames above")

# ── TASK 5: Detect Face from Digital ID ──────────────────────────────────────
dl_source_url = 'https://raw.githubusercontent.com/udacity/cd0461-building-computer-vision-solutions-with-azure-exercises/main/resources/ca-dl-sample.png'

def show_image_in_cell(face_url):
    response = requests.get(face_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()

show_image_in_cell(dl_source_url)

dl_faces = face_client.detect(
    url=dl_source_url,
    detection_model=FaceDetectionModel.DETECTION03,
    recognition_model=FaceRecognitionModel.RECOGNITION04,
    return_face_id=True
)

for face in dl_faces:
    print(f'\nFace ID {face.face_id} found in DL image')
    ids['ca-dl-sample.png'] = face.face_id

# ── Verify DL Face Against Video Thumbnail ────────────────────────────────────
dl_verify_result = face_client.verify_face_to_face(
    face_id1=ids[face_image_1],
    face_id2=ids['ca-dl-sample.png']
)

if dl_verify_result.is_identical:
    print(f"DL matches video face ✓ — confidence: {dl_verify_result.confidence}")
else:
    print(f"DL does not match video face ✗ — confidence: {dl_verify_result.confidence}")

# ── Draw Face Rectangles on DL Image ─────────────────────────────────────────
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

# ── Identify DL Face Against Person Group ────────────────────────────────────
dl_face_id = ids['ca-dl-sample.png']

identify_results = face_client.identify_from_large_person_group(
    face_ids=[dl_face_id],
    large_person_group_id=person_group_id
)

for result in identify_results:
    for candidate in result.candidates:
        print(f"Identity match confidence: {candidate.confidence}")
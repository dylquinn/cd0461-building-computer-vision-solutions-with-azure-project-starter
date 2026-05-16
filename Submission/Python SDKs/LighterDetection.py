import requests
import os, time, uuid
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials

# ── Credentials ───────────────────────────────────────────────────────────────
TRAINING_ENDPOINT     = "ENTER TRAINING ENDPOINT HERE"
training_key          = "ENTER TRAINING RESOURCE KEY HERE"
training_resource_id  = "ENTER TRAINING RESOURCE ID HERE"
PREDICTION_ENDPOINT   = "ENTER PREDICTION ENDPOINT HERE"  # Note: endpoint, not key
prediction_key        = "ENTER PREDICTION RESOURCE KEY HERE"
prediction_resource_id = "ENTER PREDICTION RESOURCE ID HERE"

training_credentials   = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer                = CustomVisionTrainingClient(TRAINING_ENDPOINT, training_credentials)

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor              = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)

# ── TASK 2 (Optional): Create Project via API ─────────────────────────────────
# Find the object detection domain
obj_detection_domain = next(
    domain for domain in trainer.get_domains()
    if domain.type == "ObjectDetection" and domain.name == "General"
)

# Create a new project
project_name = uuid.uuid4()
project = trainer.create_project(project_name, domain_id=obj_detection_domain.id)
print("Project created:", project.as_dict())
print("Project status:", project.status)

# Create tag
lighter_tag = trainer.create_tag(project.id, "Lighter")
print("Tag created: Lighter")

# ── TASK 3: Train the Model ───────────────────────────────────────────────────
# Note: if training via the UI at customvision.ai, you can skip to
# get_iterations() below and just reference your existing project ID

iteration = trainer.train_project(project.id)
while iteration.status != "Completed":
    iteration = trainer.get_iteration(project.id, iteration.id)
    print("Training status: " + iteration.status)
    print("Waiting 10 seconds...")
    time.sleep(10)

print("Training complete!")
print(iteration.as_dict())

# ── TASK 4 & 5: Validation — Precision and Recall ────────────────────────────
iteration_list = trainer.get_iterations(project.id)
for iteration_item in iteration_list:
    print(iteration_item)

model_perf = trainer.get_iteration_performance(project.id, iteration_list[0].id)
print("\nModel Performance:")
print(model_perf.as_dict())

# Explicitly print precision and recall for the task requirement
print("\nPrecision: {0:.2f}%".format(model_perf.precision * 100))
print("Recall:    {0:.2f}%".format(model_perf.recall * 100))
print("mAP:       {0:.2f}%".format(model_perf.average_precision * 100))

# Per-tag performance breakdown
for tag_perf in model_perf.per_tag_performance:
    print("\nTag: {}".format(tag_perf.name))
    print("  Precision: {0:.2f}%".format(tag_perf.precision * 100))
    print("  Recall:    {0:.2f}%".format(tag_perf.recall * 100))
    print("  mAP:       {0:.2f}%".format(tag_perf.average_precision * 100))

# ── TASK 6: Publish / Deploy to Endpoint ─────────────────────────────────────
publish_iteration_name = "udacity-2-classes-object-detection-custom"

trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print("\nModel published!")
print("Endpoint URL: {}/customvision/v3.0/Prediction/{}/detect/iterations/{}/image".format(
    PREDICTION_ENDPOINT, project.id, publish_iteration_name
))

# ── Helper: display image from URL ───────────────────────────────────────────
def show_image_from_url(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(20, 10))
    plt.imshow(img)
    plt.show()

# ── TASK 7 & 8: Predict on Test Images from GitHub ───────────────────────────
# Base URL for the project starter images on GitHub
# Update this to the correct path in the Udacity repo
BASE_IMAGE_URL = "https://raw.githubusercontent.com/dylquinn/cd0461-building-computer-vision-solutions-with-azure-project-starter/master/starter/lighter_test_images/"

test_images = [
    "lighter_test_set_1of5.jpg",
    "lighter_test_set_2of5.jpg",
    "lighter_test_set_3of5.jpg",
    "lighter_test_set_4of5.jpg",
    "lighter_test_set_5of5.jpg",
]

def perform_prediction_from_url(image_url):
    print(f"\nRunning prediction on: {image_url}")
    response = requests.get(image_url)
    image_data = response.content

    results = predictor.detect_image(project.id, publish_iteration_name, image_data)

    for prediction in results.predictions:
        print("\t{}: {:.2f}% — bounding box: left={:.3f}, top={:.3f}, width={:.3f}, height={:.3f}".format(
            prediction.tag_name,
            prediction.probability * 100,
            prediction.bounding_box.left,
            prediction.bounding_box.top,
            prediction.bounding_box.width,
            prediction.bounding_box.height
        ))
    return results

# Run predictions and display all 5 test images
for image_name in test_images:
    image_url = BASE_IMAGE_URL + image_name
    results   = perform_prediction_from_url(image_url)
    show_image_from_url(image_url)
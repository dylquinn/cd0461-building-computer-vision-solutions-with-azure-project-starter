import os
from dotenv import load_dotenv
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import DocumentAnalysisClient, DocumentModelAdministrationClient
from azure.core.credentials import AzureKeyCredential

# ── Load Environment Variables ────────────────────────────────────────────────
load_dotenv()

FORM_RECOGNIZER_ENDPOINT = os.getenv("FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY      = os.getenv("FORM_RECOGNIZER_KEY")
TRAINING_DATA_SAS_URL    = os.getenv("TRAINING_DATA_SAS_URL")
BOARDING_PASS_MODEL_ID   = os.getenv("BOARDING_PASS_MODEL_ID")

# ── Clients ───────────────────────────────────────────────────────────────────
admin_client = DocumentModelAdministrationClient(
    endpoint=FORM_RECOGNIZER_ENDPOINT,
    credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
)

analysis_client = DocumentAnalysisClient(
    endpoint=FORM_RECOGNIZER_ENDPOINT,
    credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
)

# ── Delete old model if it exists ─────────────────────────────────────────────
try:
    admin_client.delete_document_model(model_id=BOARDING_PASS_MODEL_ID)
    print(f"Old model '{BOARDING_PASS_MODEL_ID}' deleted")
except ResourceNotFoundError:
    print(f"No existing model named '{BOARDING_PASS_MODEL_ID}' — proceeding to training")

# ── Train new model ───────────────────────────────────────────────────────────
print("\nStarting training...")
poller = admin_client.begin_build_document_model(
    build_mode="template",
    blob_container_url=TRAINING_DATA_SAS_URL,
    model_id=BOARDING_PASS_MODEL_ID
)
custom_model = poller.result()

print("\n── Training Complete ──")
print(f"Model ID:    {custom_model.model_id}")
print(f"API Version: {custom_model.api_version}")
print(f"Description: {custom_model.description}")

# ── Recognized fields ─────────────────────────────────────────────────────────
print("\nRecognized fields:")
for doc_type, doc_details in custom_model.doc_types.items():
    print(f"Doc type: {doc_type}")
    for field_name, field_schema in doc_details.field_schema.items():
        confidence = doc_details.field_confidence.get(field_name, "N/A")
        print(f"  Field '{field_name}' - confidence: {confidence}")

# ── Model info ────────────────────────────────────────────────────────────────
custom_model_info = admin_client.get_document_model(model_id=custom_model.model_id)
print(f"\nModel ID:    {custom_model_info.model_id}")
print(f"Created on:  {custom_model_info.created_on}")
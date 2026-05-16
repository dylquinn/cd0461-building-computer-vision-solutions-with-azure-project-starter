
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import DocumentAnalysisClient, DocumentModelAdministrationClient
from azure.core.credentials import AzureKeyCredential

AZURE_FORM_RECOGNIZER_ENDPOINT = "INSERT ENDPOINT HERE"
AZURE_FORM_RECOGNIZER_KEY = "INSERT KEY HERE"

endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY

# Admin client for training
admin_client = DocumentModelAdministrationClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# Analysis client for extracting/testing
analysis_client = DocumentAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

trainingDataUrl = "https://apkstoragedq.blob.core.windows.net/trainingdataboardingpass?sp=racwdl&st=2026-05-12T00:35:20Z&se=2026-05-13T08:50:20Z&spr=https&sv=2025-11-05&sr=c&sig=3oqSdu7lZvEn%2F4EOL1fbtz865I2ru7T9oyjrheKDMyg%3D"

# Delete the existing model first
admin_client.delete_document_model(model_id="boarding-pass-model")
print("Old model deleted")

# Then run training again
poller = admin_client.begin_build_document_model(
    build_mode="template",
    blob_container_url=trainingDataUrl,
    model_id="boarding-pass-model"
)
custom_model = poller.result()
custom_model = poller.result()

print("Model ID:", custom_model.model_id)
print("API Version:", custom_model.api_version)
print("Description:", custom_model.description)

# Print recognized fields from the model
print("\nRecognized fields:")
for doc_type, doc_details in custom_model.doc_types.items():
    print("Doc type: {}".format(doc_type))
    for field_name, field_schema in doc_details.field_schema.items():
        confidence = doc_details.field_confidence.get(field_name, "N/A")
        print("  Field '{}' - confidence: {}".format(field_name, confidence))

# Get model info
custom_model_info = admin_client.get_document_model(model_id=custom_model.model_id)
print("\nModel ID: {}".format(custom_model_info.model_id))
print("Created on: {}".format(custom_model_info.created_on))

# Test the model on a boarding pass
new_test_url = "https://apkstoragedq.blob.core.windows.net/trainingdataboardingpass/boarding-james-webb.pdf?sp=r&st=2026-05-12T00:37:59Z&se=2026-05-13T08:52:59Z&spr=https&sv=2025-11-05&sr=b&sig=bRR9cfYzsXVdHFpm7kp31Qimf%2BnHUjJAac2D2LpQbUU%3D"

poller = analysis_client.begin_analyze_document_from_url(
    model_id=custom_model_info.model_id,
    document_url=new_test_url
)
result = poller.result()

for doc in result.documents:
    print("\nForm type: {}".format(doc.doc_type))
    for name, field in doc.fields.items():
        print("Field '{}' has value '{}' with confidence {}".format(
            name,
            field.value if field.value else field.content,
            field.confidence
        ))
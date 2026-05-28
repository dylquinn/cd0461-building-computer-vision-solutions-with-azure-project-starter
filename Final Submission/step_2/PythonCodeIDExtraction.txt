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

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

AZURE_FORM_RECOGNIZER_ENDPOINT = "ENTER FORM RECOGNIZER ENDPOINT"
AZURE_FORM_RECOGNIZER_KEY = "ENTER FORM RECOGNIZER KEY"

endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY

# Updated client
analysis_client = DocumentAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

content_url = "https://raw.githubusercontent.com/udacity/cd0461-building-computer-vision-solutions-with-azure-exercises/main/resources/ca-dl-sample.png"

# Updated method
poller = analysis_client.begin_analyze_document_from_url(
    model_id="prebuilt-idDocument",
    document_url=content_url
)
collected_id_cards = poller.result()

def get_id_card_details(identity_card):
    first_name = identity_card.fields.get("FirstName")
    if first_name:
        print("First Name: {} has confidence: {}".format(first_name.value, first_name.confidence))

    last_name = identity_card.fields.get("LastName")
    if last_name:
        print("Last Name: {} has confidence: {}".format(last_name.value, last_name.confidence))

    document_number = identity_card.fields.get("DocumentNumber")
    if document_number:
        print("Document Number: {} has confidence: {}".format(document_number.value, document_number.confidence))

    dob = identity_card.fields.get("DateOfBirth")
    if dob:
        print("Date of Birth: {} has confidence: {}".format(dob.value, dob.confidence))

    doe = identity_card.fields.get("DateOfExpiration")
    if doe:
        print("Date of Expiration: {} has confidence: {}".format(doe.value, doe.confidence))

    sex = identity_card.fields.get("Sex")
    if sex:
        print("Sex: {} has confidence: {}".format(sex.value, sex.confidence))

    address = identity_card.fields.get("Address")
    if address:
        print("Address: {} has confidence: {}".format(address.value, address.confidence))

    country_region = identity_card.fields.get("CountryRegion")
    if country_region:
        print("Country/Region: {} has confidence: {}".format(country_region.value, country_region.confidence))

    region = identity_card.fields.get("Region")
    if region:
        print("Region: {} has confidence: {}".format(region.value, region.confidence))

# Test on first result
get_id_card_details(collected_id_cards.documents[0])

# Loop through all results
for index_id, id_card in enumerate(collected_id_cards.documents):
    print("Displaying identity card details ....... # {}".format(index_id + 1))
    get_id_card_details(id_card)
    print("---------------- EOL -------------------------")
import os
import pandas as pd
import requests
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# ── Load Environment Variables ────────────────────────────────────────────────
load_dotenv()

FORM_RECOGNIZER_ENDPOINT  = os.getenv("FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY       = os.getenv("FORM_RECOGNIZER_KEY")
FACE_ENDPOINT             = os.getenv("FACE_ENDPOINT")
FACE_KEY                  = os.getenv("FACE_KEY")
PREDICTION_ENDPOINT       = os.getenv("PREDICTION_ENDPOINT")
PREDICTION_KEY            = os.getenv("PREDICTION_KEY")
PREDICTION_PROJECT_ID     = os.getenv("PREDICTION_PROJECT_ID")
PREDICTION_ITERATION_NAME = os.getenv("PREDICTION_ITERATION_NAME")
BOARDING_PASS_MODEL_ID    = os.getenv("BOARDING_PASS_MODEL_ID")
FACE_IMAGE_PATH = os.getenv("FACE_IMAGE_PATH")

# ── Clients ───────────────────────────────────────────────────────────────────
doc_client  = DocumentAnalysisClient(FORM_RECOGNIZER_ENDPOINT, AzureKeyCredential(FORM_RECOGNIZER_KEY))
face_client = FaceClient(FACE_ENDPOINT, AzureKeyCredential(FACE_KEY))
predictor   = CustomVisionPredictionClient(
    PREDICTION_ENDPOINT,
    ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
)

# ── Load Manifest ─────────────────────────────────────────────────────────────
manifest = pd.read_csv(os.getenv("MANIFEST_PATH"))

# Strip whitespace from all column names to avoid issues with leading/trailing spaces
manifest.columns = manifest.columns.str.strip()

print("Columns:", manifest.columns.tolist())
print("First row:", manifest.iloc[0].tolist())

# Set all validation columns to False at the start
manifest["Passenger Name Validation"]              = False
manifest["Passenger Date of Birth Validation"]     = False
manifest["Passenger Face Validation"]              = False
manifest["Passenger Flight Details Validation"]    = False
manifest["Passenger Carry-on Baggage Validation"]  = False

# ─────────────────────────────────────────────────────────────────────────────
# BASE URLs — all pulled from GitHub, no Blob Storage needed
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_BASE  = "https://raw.githubusercontent.com/dylquinn/cd0461-building-computer-vision-solutions-with-azure-project-starter/master/starter"
BP_BASE      = f"{GITHUB_BASE}/boarding_pass_template"
DL_BASE      = f"{GITHUB_BASE}/digital_id_template"
LIGHTER_BASE = f"{GITHUB_BASE}/lighter_test_images"

# ─────────────────────────────────────────────────────────────────────────────
# PASSENGER ASSETS
# ─────────────────────────────────────────────────────────────────────────────
passenger_assets = {
    "AVKASH CHAUHAN": {
        "id_image_url":      f"{DL_BASE}/ca-dl-avkash-chauhan.png",
        "boarding_pass_url": f"{BP_BASE}/boarding-avkash.pdf",
        "video_face_path":   FACE_IMAGE_PATH,
        "luggage_image_url": f"{LIGHTER_BASE}/lighter_test_set_1of5.jpg",
    },
    "JAMES JACKSON": {
        "id_image_url":      f"{DL_BASE}/ca-dl-james-jackson.png",
        "boarding_pass_url": f"{BP_BASE}/boarding-james.pdf",
        "video_face_path":   FACE_IMAGE_PATH,
        "luggage_image_url": f"{LIGHTER_BASE}/lighter_test_set_2of5.jpg",
    },
    "JAMES WEBB": {
        "id_image_url":      f"{DL_BASE}/ca-dl-james-webb.png",
        "boarding_pass_url": f"{BP_BASE}/boarding-james-webb.pdf",
        "video_face_path":   FACE_IMAGE_PATH,
        "luggage_image_url": f"{LIGHTER_BASE}/lighter_test_set_3of5.jpg",
    },
    "LIBBY HEROLD": {
        "id_image_url":      f"{DL_BASE}/ca-dl-libby-herold.png",
        "boarding_pass_url": f"{BP_BASE}/boarding-libby.pdf",
        "video_face_path":   FACE_IMAGE_PATH,
        "luggage_image_url": f"{LIGHTER_BASE}/lighter_test_set_4of5.jpg",
    },
    "RADHA S KUMAR": {
        "id_image_url":      f"{DL_BASE}/ca-dl-radha-s-kumar.png",
        "boarding_pass_url": f"{BP_BASE}/boarding-radha-s-kumar.pdf",
        "video_face_path":   FACE_IMAGE_PATH,
        "luggage_image_url": f"{LIGHTER_BASE}/lighter_test_set_5of5.jpg",
    },
    "SAMEER KUMAR": {
        "id_image_url":      f"{DL_BASE}/ca-dl-sameer-kumar.png",
        "boarding_pass_url": f"{BP_BASE}/boarding-sameer.pdf",
        "video_face_path":   FACE_IMAGE_PATH,
        "luggage_image_url": f"{LIGHTER_BASE}/lighter_test_set_1of5.jpg",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2A — Extract from Digital ID
# ─────────────────────────────────────────────────────────────────────────────
def extract_id_data(id_image_url):
    poller = doc_client.begin_analyze_document_from_url("prebuilt-idDocument", id_image_url)
    result = poller.result()
    doc    = result.documents[0]
    return {
        "first_name": doc.fields.get("FirstName").value        if doc.fields.get("FirstName")  else "",
        "last_name":  doc.fields.get("LastName").value         if doc.fields.get("LastName")   else "",
        "dob":        str(doc.fields.get("DateOfBirth").value) if doc.fields.get("DateOfBirth") else "",
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2B — Extract from Boarding Pass
# ─────────────────────────────────────────────────────────────────────────────
def extract_boarding_pass_data(bp_url):
    poller = doc_client.begin_analyze_document_from_url(BOARDING_PASS_MODEL_ID, bp_url)
    result = poller.result()
    doc    = result.documents[0]
    def get(field):
        f = doc.fields.get(field)
        return f.value if f else ""
    return {
        "passenger_name": get("PassengerName"),
        "flight_number":  get("FlightNumber"),
        "seat":           get("Seat"),
        "class":          get("Class"),
        "origin":         get("FromCity"),
        "destination":    get("ToCity"),
        "date":           get("Date"),
        "boarding_time":  get("BoardingTime"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Face Match (ID photo vs video thumbnail)
# ─────────────────────────────────────────────────────────────────────────────
def get_face_match_confidence(id_image_url, video_face_path):
    # Detect face from ID image using URL method
    id_faces = face_client.detect_from_url(
        url=id_image_url,
        detection_model=FaceDetectionModel.DETECTION03,
        recognition_model=FaceRecognitionModel.RECOGNITION04,
        return_face_id=True
    )
    if not id_faces:
        print("  No face detected in ID image")
        return 0.0

    # Detect face from local video thumbnail using stream method
    with open(video_face_path, "rb") as img:
        video_faces = face_client.detect(
            image_content=img.read(),
            detection_model=FaceDetectionModel.DETECTION03,
            recognition_model=FaceRecognitionModel.RECOGNITION04,
            return_face_id=True
        )
    if not video_faces:
        print("  No face detected in video thumbnail")
        return 0.0

    result = face_client.verify_face_to_face(
        face_id1=id_faces[0].face_id,
        face_id2=video_faces[0].face_id
    )
    print(f"  Face match confidence: {result.confidence:.2f} — identical: {result.is_identical}")
    return result.confidence if result.is_identical else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Lighter Detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_lighter(luggage_image_url):
    response   = requests.get(luggage_image_url)
    image_data = response.content
    results    = predictor.detect_image(PREDICTION_PROJECT_ID, PREDICTION_ITERATION_NAME, image_data)
    for prediction in results.predictions:
        if prediction.tag_name == "Lighter" and prediction.probability > 0.5:
            print(f"  Lighter detected with confidence: {prediction.probability:.2f}")
            return True
    print("  No lighter detected above threshold")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# KIOSK MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
def generate_kiosk_message(row, lighter_found):
    name          = str(row.get("Passanger Name", "")).strip()
    flight_number = str(row.get("Flight No.", "N/A")).strip()
    boarding_time = str(row.get("Boarding Time", "N/A")).strip()
    origin        = str(row.get("From", "N/A")).strip()
    destination   = str(row.get("To", "N/A")).strip()
    seat          = str(row.get("Seat", "N/A")).strip()

    name_valid    = row["Passenger Name Validation"]
    dob_valid     = row["Passenger Date of Birth Validation"]
    bp_valid      = row["Passenger Flight Details Validation"]
    person_valid  = row["Passenger Face Validation"]
    luggage_valid = row["Passenger Carry-on Baggage Validation"]

    passed = sum([name_valid, dob_valid, bp_valid, person_valid, luggage_valid])

    print("\n" + "="*60)
    print("KIOSK SCREEN MESSAGE")
    print("="*60)

    if not bp_valid:
        print("""
Dear Sir/Madam,

Some of the information in your boarding pass does not match
the flight manifest data, so you cannot board the plane.

Please see a customer service representative.
        """)
        return

    if not name_valid or not dob_valid:
        print("""
Dear Sir/Madam,

Some of the information on your ID card does not match
the flight manifest data, so you cannot board the plane.

Please see a customer service representative.
        """)
        return

    print(f"""
Dear {name},

You are welcome to flight #{flight_number} leaving at {boarding_time}
from {origin} to {destination}.
Your seat number is {seat}, and it is confirmed.
    """)

    if lighter_found:
        print("We have found a prohibited item in your carry-on baggage,")
        print("and it is flagged for removal.\n")
    else:
        print("We did not find a prohibited item (lighter) in your carry-on baggage,")
        print("thanks for following the procedure.\n")

    if person_valid:
        print("Your identity is verified so please board the plane.")
    else:
        print("Your identity could not be verified.")
        print("Please see a customer service representative.")

    print(f"\nValidations passed: {passed}/5")
    print("STATUS: CLEARED TO BOARD ✓" if passed >= 3 else "STATUS: REFER TO CUSTOMER SERVICE ✗")
    print("="*60)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — Process each passenger
# ─────────────────────────────────────────────────────────────────────────────
for index, row in manifest.iterrows():
    # Use the "Passanger Name" column (note typo in CSV is intentional)
    full_name = str(row.get("Passanger Name", "")).strip().upper()
    print(f"\nProcessing passenger: {full_name}")

    assets = passenger_assets.get(full_name)
    if not assets:
        print(f"  No assets found for '{full_name}' — skipping")
        continue

    # Run all extractions
    id_data    = extract_id_data(assets["id_image_url"])
    bp_data    = extract_boarding_pass_data(assets["boarding_pass_url"])
    confidence = get_face_match_confidence(assets["id_image_url"], assets["video_face_path"])
    lighter    = detect_lighter(assets["luggage_image_url"])

    # Name Validation — 3 way match: manifest, ID, boarding pass
    manifest_name = full_name
    id_name       = f"{id_data['first_name']} {id_data['last_name']}".strip().upper()
    bp_name       = bp_data["passenger_name"].strip().upper()
    manifest.at[index, "Passenger Name Validation"] = (manifest_name == id_name) and (manifest_name == bp_name)

    # DoB Validation
    manifest.at[index, "Passenger Date of Birth Validation"] = (
        str(row.get("Date of Birth", "")).strip() == id_data["dob"]
    )

    # Boarding Pass Validation — match against actual CSV column names
    manifest.at[index, "Passenger Flight Details Validation"] = (
        str(row.get("Flight No.", "")).strip().upper() == bp_data["flight_number"].strip().upper() and
        str(row.get("Seat",       "")).strip().upper() == bp_data["seat"].strip().upper()          and
        str(row.get("Class",      "")).strip().upper() == bp_data["class"].strip().upper()         and
        str(row.get("From",       "")).strip().upper() == bp_data["origin"].strip().upper()        and
        str(row.get("To",         "")).strip().upper() == bp_data["destination"].strip().upper()   and
        str(row.get("Date",       "")).strip()         == bp_data["date"].strip()                  and
        str(row.get("Boarding Time", "")).strip()      == bp_data["boarding_time"].strip()
    )

    # Person Validation
    manifest.at[index, "Passenger Face Validation"] = confidence >= 0.65

    # Luggage Validation — left as FALSE per project instructions
    manifest.at[index, "Passenger Carry-on Baggage Validation"] = False

    # Kiosk Message
    generate_kiosk_message(manifest.loc[index], lighter)

# ── Save Final Manifest ───────────────────────────────────────────────────────
manifest.to_csv("manifest_validated.csv", index=False)
print("\nFinal manifest saved to manifest_validated.csv")
print("\nValidation Summary:")
print(manifest[[
    "Passanger Name",
    "Passenger Name Validation",
    "Passenger Date of Birth Validation",
    "Passenger Face Validation",
    "Passenger Flight Details Validation",
    "Passenger Carry-on Baggage Validation"
]].to_string(index=False))
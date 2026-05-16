import pandas as pd
import requests
from io import BytesIO
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# ── Credentials ───────────────────────────────────────────────────────────────
FORM_RECOGNIZER_ENDPOINT   = "ENTER ENDPOINT"
FORM_RECOGNIZER_KEY        = "ENTER KEY"
FACE_ENDPOINT              = "ENTER ENDPOINT"
FACE_KEY                   = "ENTER KEY"
PREDICTION_ENDPOINT        = "ENTER ENDPOINT"
PREDICTION_KEY             = "ENTER KEY"
PREDICTION_PROJECT_ID      = "ENTER PROJECT ID"
PREDICTION_ITERATION_NAME  = "udacity-2-classes-object-detection-custom"
BOARDING_PASS_MODEL_ID     = "boarding-pass-model"

# ── Clients ───────────────────────────────────────────────────────────────────
doc_client  = DocumentAnalysisClient(FORM_RECOGNIZER_ENDPOINT, AzureKeyCredential(FORM_RECOGNIZER_KEY))
face_client = FaceClient(FACE_ENDPOINT, AzureKeyCredential(FACE_KEY))
predictor   = CustomVisionPredictionClient(
    PREDICTION_ENDPOINT,
    ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
)

# ── Load Manifest ─────────────────────────────────────────────────────────────
manifest = pd.read_csv("manifest.csv")
manifest["NameValidation"]         = False
manifest["DoBValidation"]          = False
manifest["PersonValidation"]       = False
manifest["BoardingPassValidation"] = False
manifest["LuggageValidation"]      = False

# ─────────────────────────────────────────────────────────────────────────────
# PASSENGER ASSETS
# Map each passenger name to their associated files.
# Update the SAS URLs and face image paths for each passenger.
# video_face_path: local .jpg file saved from Script 1 (human-face1.jpg etc.)
# ─────────────────────────────────────────────────────────────────────────────
passenger_assets = {
    "JAMES HARTWELL": {
        "id_image_url":      "ENTER SAS URL TO JAMES ID IMAGE",
        "boarding_pass_url": "ENTER SAS URL TO JAMES BOARDING PASS PDF",
        "video_face_path":   "human-face1.jpg",
        "luggage_image_url": "ENTER SAS URL TO JAMES LUGGAGE IMAGE",
    },
    "SOFIA MARTINEZ": {
        "id_image_url":      "ENTER SAS URL TO SOFIA ID IMAGE",
        "boarding_pass_url": "ENTER SAS URL TO SOFIA BOARDING PASS PDF",
        "video_face_path":   "human-face2.jpg",
        "luggage_image_url": "ENTER SAS URL TO SOFIA LUGGAGE IMAGE",
    },
    "CHEN WEI": {
        "id_image_url":      "ENTER SAS URL TO CHEN ID IMAGE",
        "boarding_pass_url": "ENTER SAS URL TO CHEN BOARDING PASS PDF",
        "video_face_path":   "human-face3.jpg",
        "luggage_image_url": "ENTER SAS URL TO CHEN LUGGAGE IMAGE",
    },
    "PRIYA NAIR": {
        "id_image_url":      "ENTER SAS URL TO PRIYA ID IMAGE",
        "boarding_pass_url": "ENTER SAS URL TO PRIYA BOARDING PASS PDF",
        "video_face_path":   "human-face4.jpg",
        "luggage_image_url": "ENTER SAS URL TO PRIYA LUGGAGE IMAGE",
    },
    "LUCAS DUPONT": {
        "id_image_url":      "ENTER SAS URL TO LUCAS ID IMAGE",
        "boarding_pass_url": "ENTER SAS URL TO LUCAS BOARDING PASS PDF",
        "video_face_path":   "human-face5.jpg",
        "luggage_image_url": "ENTER SAS URL TO LUCAS LUGGAGE IMAGE",
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
def extract_boarding_pass_data(bp_image_url):
    poller = doc_client.begin_analyze_document_from_url(BOARDING_PASS_MODEL_ID, bp_image_url)
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
    # Detect face from ID image
    id_faces = face_client.detect(
        url=id_image_url,
        detection_model=FaceDetectionModel.DETECTION03,
        recognition_model=FaceRecognitionModel.RECOGNITION04,
        return_face_id=True
    )
    if not id_faces:
        print("  No face detected in ID image")
        return 0.0

    # Detect face from saved video thumbnail
    with open(video_face_path, "rb") as img:
        video_faces = face_client.detect(
            image_content=img,
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
    return False

# ─────────────────────────────────────────────────────────────────────────────
# KIOSK MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
def generate_kiosk_message(row, lighter_found):
    name          = f"{row.get('FirstName', '')} {row.get('LastName', '')}".strip()
    flight_number = row.get("FlightNumber", "N/A")
    boarding_time = row.get("BoardingTime", "N/A")
    origin        = row.get("Origin",      "N/A")
    destination   = row.get("Destination", "N/A")
    seat          = row.get("Seat",        "N/A")

    name_valid    = row["NameValidation"]
    dob_valid     = row["DoBValidation"]
    bp_valid      = row["BoardingPassValidation"]
    person_valid  = row["PersonValidation"]
    luggage_valid = row["LuggageValidation"]

    passed = sum([name_valid, dob_valid, bp_valid, person_valid, luggage_valid])

    print("\n" + "="*60)
    print("KIOSK SCREEN MESSAGE")
    print("="*60)

    # Boarding pass mismatch
    if not bp_valid:
        print("""
Dear Sir/Madam,

Some of the information in your boarding pass does not match
the flight manifest data, so you cannot board the plane.

Please see a customer service representative.
        """)
        return

    # ID mismatch
    if not name_valid or not dob_valid:
        print("""
Dear Sir/Madam,

Some of the information on your ID card does not match
the flight manifest data, so you cannot board the plane.

Please see a customer service representative.
        """)
        return

    # Flight details confirmed
    print(f"""
Dear {name},

You are welcome to flight #{flight_number} leaving at {boarding_time}
from {origin} to {destination}.
Your seat number is {seat}, and it is confirmed.
    """)

    # Luggage message
    if lighter_found:
        print("We have found a prohibited item in your carry-on baggage,")
        print("and it is flagged for removal.\n")
    else:
        print("We did not find a prohibited item (lighter) in your carry-on baggage,")
        print("thanks for following the procedure.\n")

    # Identity message
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
    full_name = f"{row.get('FirstName', '')} {row.get('LastName', '')}".strip().upper()
    print(f"\nProcessing passenger: {full_name}")

    assets = passenger_assets.get(full_name)
    if not assets:
        print(f"  No assets found for {full_name} — skipping")
        continue

    # Run all extractions
    id_data    = extract_id_data(assets["id_image_url"])
    bp_data    = extract_boarding_pass_data(assets["boarding_pass_url"])
    confidence = get_face_match_confidence(assets["id_image_url"], assets["video_face_path"])
    lighter    = detect_lighter(assets["luggage_image_url"])

    # ── Name Validation ───────────────────────────────────────────────────────
    manifest_name = full_name
    id_name       = f"{id_data['first_name']} {id_data['last_name']}".strip().upper()
    bp_name       = bp_data["passenger_name"].strip().upper()
    manifest.at[index, "NameValidation"] = (manifest_name == id_name) and (manifest_name == bp_name)

    # ── DoB Validation ────────────────────────────────────────────────────────
    manifest.at[index, "DoBValidation"] = str(row.get("DateOfBirth", "")).strip() == id_data["dob"]

    # ── Boarding Pass Validation ──────────────────────────────────────────────
    manifest.at[index, "BoardingPassValidation"] = (
        str(row.get("FlightNumber",  "")).strip().upper() == bp_data["flight_number"].strip().upper()  and
        str(row.get("Seat",          "")).strip().upper() == bp_data["seat"].strip().upper()           and
        str(row.get("Class",         "")).strip().upper() == bp_data["class"].strip().upper()          and
        str(row.get("Origin",        "")).strip().upper() == bp_data["origin"].strip().upper()         and
        str(row.get("Destination",   "")).strip().upper() == bp_data["destination"].strip().upper()    and
        str(row.get("FlightDate",    "")).strip()         == bp_data["date"].strip()                   and
        str(row.get("BoardingTime",  "")).strip()         == bp_data["boarding_time"].strip()
    )

    # ── Person Validation ─────────────────────────────────────────────────────
    manifest.at[index, "PersonValidation"] = confidence >= 0.65

    # ── Luggage Validation ────────────────────────────────────────────────────
    # Left as FALSE per project instructions — no way to match luggage to passenger
    manifest.at[index, "LuggageValidation"] = False

    # ── Kiosk Message ─────────────────────────────────────────────────────────
    generate_kiosk_message(manifest.loc[index], lighter)

# ── Save Final Manifest ───────────────────────────────────────────────────────
manifest.to_csv("manifest_validated.csv", index=False)
print("\nFinal manifest saved to manifest_validated.csv")
print("\nValidation Summary:")
print(manifest[[
    "FirstName",
    "LastName",
    "NameValidation",
    "DoBValidation",
    "BoardingPassValidation",
    "PersonValidation",
    "LuggageValidation"
]].to_string(index=False))
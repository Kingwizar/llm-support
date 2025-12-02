import cv2
import mediapipe as mp
import face_recognition
import os
import json
import numpy as np

# ================================
#    INITIALISATION DOSSIERS
# ================================
DB_DIR = "face_db"
os.makedirs(DB_DIR, exist_ok=True)

DB_FILE = "database.json"

# Charger la base existante
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        database = json.load(f)
else:
    database = {}  # { "person_001": "encodage_path" }

# Charger les encodages
known_encodings = []
known_ids = []

for person_id in database:
    enc_path = database[person_id]["encoding"]
    if os.path.exists(enc_path):
        enc = np.load(enc_path)
        known_encodings.append(enc)
        known_ids.append(person_id)


# ================================
#       MEDIAPIPE DETECTOR
# ================================
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ================================
#        CAPTURE CAMERA
# ================================
cap = cv2.VideoCapture(0)

def generate_new_id():
    """Création ID style: person_001"""
    existing = [int(pid.split("_")[1]) for pid in database.keys()] or [0]
    new_id_num = max(existing) + 1
    return f"person_{new_id_num:03d}"


print("📷 Système actif — reconnaissance + ajout automatique\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if result.detections:
        for detection in result.detections:

            # BOUNDING BOX
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)
            x2 = x1 + w_box
            y2 = y1 + h_box

            face_crop = rgb[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            encodings = face_recognition.face_encodings(face_crop)

            face_id = "INCONNU"

            if len(encodings) > 0:
                enc = encodings[0]

                # COMPARAISON
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.45)
                face_distances = face_recognition.face_distance(known_encodings, enc)

                if len(face_distances) > 0:
                    best_match = np.argmin(face_distances)

                    if matches[best_match]:
                        face_id = known_ids[best_match]

                    else:
                        # Nouveau visage → création d’un ID + sauvegarde
                        new_id = generate_new_id()
                        person_dir = os.path.join(DB_DIR, new_id)
                        os.makedirs(person_dir, exist_ok=True)

                        # Sauvegarder l’image
                        img_path = os.path.join(person_dir, "face.jpg")
                        cv2.imwrite(img_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

                        # Sauvegarder encodage
                        enc_path = os.path.join(person_dir, "encoding.npy")
                        np.save(enc_path, enc)

                        # Mise à jour DB
                        database[new_id] = {
                            "image": img_path,
                            "encoding": enc_path
                        }

                        with open(DB_FILE, "w") as f:
                            json.dump(database, f, indent=4)

                        known_encodings.append(enc)
                        known_ids.append(new_id)

                        print(f"[NOUVEAU VISAGE] → {new_id}")
                        face_id = new_id

                else:
                    # Si aucun encodage existant → ajouter direct
                    new_id = generate_new_id()
                    person_dir = os.path.join(DB_DIR, new_id)
                    os.makedirs(person_dir, exist_ok=True)

                    img_path = os.path.join(person_dir, "face.jpg")
                    cv2.imwrite(img_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                    enc_path = os.path.join(person_dir, "encoding.npy")
                    np.save(enc_path, enc)

                    database[new_id] = {
                        "image": img_path,
                        "encoding": enc_path
                    }

                    with open(DB_FILE, "w") as f:
                        json.dump(database, f, indent=4)

                    known_encodings.append(enc)
                    known_ids.append(new_id)

                    print(f"[NOUVEAU VISAGE] → {new_id}")
                    face_id = new_id

            # Affichage ID au dessus
            cv2.putText(frame, face_id, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            mp_draw.draw_detection(frame, detection)

    cv2.imshow("Face Recognition Auto-DB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

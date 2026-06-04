from pathlib import Path
from functools import lru_cache

import cv2
import face_recognition
import numpy as np
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode

full_path_haarcascade = cv2.__path__[0] + "/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(full_path_haarcascade)
full_path_eyecascade = cv2.__path__[0] + "/data/haarcascade_eye_tree_eyeglasses.xml"
eye_cascade = cv2.CascadeClassifier(full_path_eyecascade)
imagen = cv2.imread("image.png")

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MODEL_PATH = Path("documentos") / "models" / "face_landmarker.task"
DOCUMENTOS_DIR = Path("documentos")
BLACKLIST_DIR = Path("blacklist")
KNOWN_FACE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
KNOWN_FACE_TOLERANCE = 0.5
BLINK_BLENDSHAPE_THRESHOLD = 0.35
BLINK_EAR_THRESHOLD = 0.21
CLOSED_EYE_MIN_FRAMES = 2


def _load_face_landmarker():
    if not MODEL_PATH.exists():
        return None

    try:
        base_options = mp_base_options.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = face_landmarker.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_running_mode.VisionTaskRunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1,
        )
        return face_landmarker.FaceLandmarker.create_from_options(options)
    except Exception:
        return None


def _euclidean_distance(point_a, point_b):
    return np.linalg.norm(np.array(point_a) - np.array(point_b))


def _eye_aspect_ratio(landmarks, indices, image_width, image_height):
    points = [
        (landmarks[index].x * image_width, landmarks[index].y * image_height)
        for index in indices
    ]

    vertical_one = _euclidean_distance(points[1], points[5])
    vertical_two = _euclidean_distance(points[2], points[4])
    horizontal = _euclidean_distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_one + vertical_two) / (2.0 * horizontal)


def _iter_registered_face_images():
    if not DOCUMENTOS_DIR.exists():
        return

    for image_path in DOCUMENTOS_DIR.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in KNOWN_FACE_EXTENSIONS:
            continue
        if "uploads" in image_path.parts or "models" in image_path.parts:
            continue
        yield image_path


def _iter_blacklisted_face_images():
    if not BLACKLIST_DIR.exists():
        return

    for image_path in BLACKLIST_DIR.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in KNOWN_FACE_EXTENSIONS:
            continue
        yield image_path


@lru_cache(maxsize=1)
def _load_registered_faces():
    known_encodings = []
    known_names = []

    for image_path in _iter_registered_face_images() or []:
        image = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(image)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(image_path.parent.name)

    return known_encodings, known_names


@lru_cache(maxsize=1)
def _load_blacklisted_faces():
    known_encodings = []
    known_names = []

    for image_path in _iter_blacklisted_face_images() or []:
        image = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(image)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(image_path.stem)

    return known_encodings, known_names


def _match_frame_against_registered_faces(frame):
    known_encodings, known_names = _load_registered_faces()
    if not known_encodings:
        return False, "NO REGISTERED FACES"

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    if not face_locations:
        return False, "NO FACE DETECTED"

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) == 0:
            continue

        best_match_index = int(np.argmin(distances))
        best_distance = float(distances[best_match_index])
        if best_distance <= KNOWN_FACE_TOLERANCE:
            return True, known_names[best_match_index]

    return False, "UNKNOWN FACE"


def _match_frame_against_blacklisted_faces(frame):
    known_encodings, known_names = _load_blacklisted_faces()
    if not known_encodings:
        return False, "NO BLACKLISTED FACES"

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    if not face_locations:
        return False, "NO FACE DETECTED"

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) == 0:
            continue

        best_match_index = int(np.argmin(distances))
        best_distance = float(distances[best_match_index])
        if best_distance <= KNOWN_FACE_TOLERANCE:
            return True, known_names[best_match_index]

    return False, "NOT BLACKLISTED"


class LivenessVerifier:
    def __init__(self):
        self.face_landmarker = _load_face_landmarker()
        self.frame_timestamp_ms = 0
        self.closed_eye_frames = 0
        self.blink_count = 0

    def verify_frame_detail(self, frame):
        annotated_frame = frame.copy()
        live = False
        face_detected = False
        left_blink = 0.0
        right_blink = 0.0
        average_blink = 0.0
        ear = 0.0
        eye_count = 0
        source = "mediapipe"

        if self.face_landmarker is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_image = mp_image.Image(mp_image.ImageFormat.SRGB, rgb_frame)
            self.frame_timestamp_ms += 33
            result = self.face_landmarker.detect_for_video(
                current_image, self.frame_timestamp_ms
            )

            if result.face_landmarks:
                face_detected = True
                landmarks = result.face_landmarks[0]
                image_height, image_width = frame.shape[:2]
                left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE, image_width, image_height)
                right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE, image_width, image_height)
                ear = (left_ear + right_ear) / 2.0

            if result.face_blendshapes:
                blendshapes = result.face_blendshapes[0]
                blendshape_scores = {
                    category.category_name: category.score or 0.0
                    for category in blendshapes
                    if category.category_name
                }

                left_blink = float(blendshape_scores.get("eyeBlinkLeft", 0.0))
                right_blink = float(blendshape_scores.get("eyeBlinkRight", 0.0))
                average_blink = (left_blink + right_blink) / 2.0

                if average_blink >= BLINK_BLENDSHAPE_THRESHOLD:
                    self.closed_eye_frames += 1
                else:
                    if self.closed_eye_frames >= CLOSED_EYE_MIN_FRAMES:
                        self.blink_count += 1
                    self.closed_eye_frames = 0
            elif face_detected:
                if ear > 0 and ear <= BLINK_EAR_THRESHOLD:
                    self.closed_eye_frames += 1
                else:
                    if self.closed_eye_frames >= CLOSED_EYE_MIN_FRAMES:
                        self.blink_count += 1
                    self.closed_eye_frames = 0
                average_blink = ear
            else:
                self.closed_eye_frames = 0
                self.blink_count = 0

            live = self.blink_count > 0
            status_text = "LIVE" if live else "LOOK AT CAMERA"
            status_color = (0, 255, 0) if live else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                f"{status_text} | blinks: {self.blink_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
                cv2.LINE_AA,
            )

            if not face_detected:
                cv2.putText(
                    annotated_frame,
                    "NO FACE DETECTED",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            source = "cascade"
            faces = face_cascade.detectMultiScale(
                frame,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(500, 500),
            )

            if len(faces) > 0:
                face_detected = True
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                roi_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(15, 15),
                )
                eye_count = len(eyes)

                if eye_count == 0:
                    self.closed_eye_frames += 1
                    left_blink = 1.0
                    right_blink = 1.0
                    average_blink = 1.0
                else:
                    if self.closed_eye_frames >= CLOSED_EYE_MIN_FRAMES:
                        self.blink_count += 1
                    self.closed_eye_frames = 0
                    left_blink = 0.0
                    right_blink = 0.0
                    average_blink = 0.0

                live = self.blink_count > 0
                status_text = "LIVE" if live else "LOOK AT CAMERA"
                status_color = (0, 255, 0) if live else (0, 0, 255)
                cv2.putText(
                    annotated_frame,
                    f"{status_text} | blinks: {self.blink_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_frame,
                    f"FACE DETECTED | eyes: {eye_count}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                self.closed_eye_frames = 0
                self.blink_count = 0
                cv2.putText(
                    annotated_frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        return annotated_frame, live, {
            "face_detected": face_detected,
            "blink_count": self.blink_count,
            "closed_eye_frames": self.closed_eye_frames,
            "live": live,
            "left_blink": left_blink,
            "right_blink": right_blink,
            "average_blink": average_blink,
            "ear": ear,
            "eye_count": eye_count,
            "source": source,
        }
    def verify_frame(self, frame):
        annotated_frame, live, _ = self.verify_frame_detail(frame)
        return annotated_frame, live


liveness_verifier = LivenessVerifier()


def procesar_imagen(filename, content):

    label = filename.split(".")[0]
    direccion = Path("documentos") / label
    direccion.mkdir(parents=True, exist_ok=True)

    nparr = np.frombuffer(content, np.uint8)
    imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if imagen is None:
        raise ValueError("No se pudo decodificar la imagen subida")

    #Proceso la imagen para detectar rostros
    faces = face_cascade.detectMultiScale(
        imagen,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(500, 500),
    )
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 0, 255), 10)
        face = imagen[y:y + h, x:x + w]

        cv2.imwrite(str(direccion / f"{label}.jpg"), face)
        cv2.imshow("imagen", imagen)
        k = cv2.waitKey(0)

    _load_registered_faces.cache_clear()


def verificar_liveliness(frame, detail=False):
    if detail:
        return liveness_verifier.verify_frame_detail(frame)

    return liveness_verifier.verify_frame(frame)


def verificar_login(frame):
    matched, label = _match_frame_against_registered_faces(frame)
    annotated_frame = frame.copy()

    status_text = f"MATCH: {label}" if matched else label
    status_color = (0, 255, 0) if matched else (0, 0, 255)

    cv2.putText(
        annotated_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return annotated_frame, matched, label


def verificar_blacklist(frame):
    matched, label = _match_frame_against_blacklisted_faces(frame)
    annotated_frame = frame.copy()

    status_text = f"BLACKLIST MATCH: {label}" if matched else label
    status_color = (0, 0, 255) if matched else (0, 255, 0)

    cv2.putText(
        annotated_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return annotated_frame, matched, label


def refresh_blacklist_cache():
    _load_blacklisted_faces.cache_clear()
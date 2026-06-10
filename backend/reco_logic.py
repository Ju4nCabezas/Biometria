from functools import lru_cache
from pathlib import Path

import cv2
import easyocr
import face_recognition
import numpy as np

import backend.LivenessVerifier as lv

full_path_haarcascade = cv2.__path__[0] + "/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(full_path_haarcascade)
full_path_eyecascade = cv2.__path__[0] + "/data/haarcascade_eye_tree_eyeglasses.xml"
eye_cascade = cv2.CascadeClassifier(full_path_eyecascade)
imagen = cv2.imread("image.png")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "documentos" / "models" / "face_landmarker.task"
DOCUMENTOS_DIR = BASE_DIR / "documentos"
BLACKLIST_DIR = BASE_DIR / "documentos" / "blacklist"

KNOWN_FACE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
KNOWN_FACE_TOLERANCE = 0.5
BLINK_BLENDSHAPE_THRESHOLD = 0.35
BLINK_EAR_THRESHOLD = 0.21
CLOSED_EYE_MIN_FRAMES = 2


@lru_cache(maxsize=1)
def _get_ocr_reader():
    return easyocr.Reader(["es"], gpu=False)


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

    # Carga y cachea los encodings faciales de todos los usuarios registrados.
    # Devuelve dos listas paralelas: encodings (vectores de 128 dimensiones) y
    # nombres (nombre de la carpeta padre de cada imagen).
    # El cache se invalida llamando a _load_registered_faces.cache_clear()
    # tras registrar o eliminar un usuario.

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


#### LO IMPORTANTE!!!!

def _match_frame_against_registered_faces(frame):  
    # Compara el frame con todos los rostros registrados.
    # 1. Convierte el frame de BGR (OpenCV) a RGB (face_recognition).
    # 2. Detecta ubicaciones de rostros con el modelo HOG.
    # 3. Calcula el encoding de cada rostro detectado.
    # 4. Calcula la distancia euclidiana contra todos los encodings conocidos.
    # 5. Si la distancia mínima está dentro del umbral, devuelve True y el nombre.

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


def _find_matching_registered_label(frame) -> str | None:
    matched, label = _match_frame_against_registered_faces(frame)
    if matched:
        return label

    return None


def _clear_registered_face_folder(folder_path: Path):
    # Elimina todas las imágenes existentes en la carpeta de un usuario registrado
    # antes de guardar la nueva. Esto evita acumular múltiples imágenes del mismo
    # usuario si se vuelve a registrar.

    if not folder_path.exists():
        return

    for image_path in folder_path.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in KNOWN_FACE_EXTENSIONS:
            image_path.unlink()


def _save_registered_face(label: str, face_image):
    # Crea (o reutiliza) la carpeta del usuario, limpia imágenes anteriores
    # y guarda el recorte del rostro como '{label}.jpg'.

    directory = DOCUMENTOS_DIR / label
    directory.mkdir(parents=True, exist_ok=True)
    _clear_registered_face_folder(directory)
    cv2.imwrite(str(directory / f"{label}.jpg"), face_image)


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



# Instancia global del verificador de liveliness. Se comparte entre llamadas
# a verificar_liveliness() fuera del flujo de login (endpoint /video).

liveness_verifier = lv.LivenessVerifier()


def procesar_imagen(filename, content):
    nombre_ocr = extraer_nombre_documento(content)
        # Extrae el nombre del documento via OCR. Si no se puede extraer, lanza
    # excepción — el registro no puede continuar sin nombre válido.

    if not nombre_ocr:
        raise ValueError("No se pudo extraer nombre y apellido con OCR")

    # Decodifica los bytes de la imagen a un array de NumPy que OpenCV pueda leer.
    nparr = np.frombuffer(content, np.uint8)
    imagen_local = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if imagen_local is None:
        raise ValueError("No se pudo decodificar la imagen subida")

    # Detecta rostros en la imagen del documento con el clasificador Haar.

    faces = face_cascade.detectMultiScale(
        imagen_local,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(500, 500),
    )
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(imagen_local, (x, y), (x + w, y + h), (0, 0, 255), 10)
        # Recorta el rostro detectado del documento.
        face = imagen_local[y : y + h, x : x + w]

        # Si el rostro ya existe en el sistema con otro nombre (re-registro),
        # se reutiliza el label conocido para no duplicar entradas. Si es nuevo,
        # se usa el nombre extraído por OCR.

        matched_label = _find_matching_registered_label(face)
        label = matched_label if matched_label else nombre_ocr
        _save_registered_face(label, face)
        
    # Invalida el cache de encodings para que el nuevo usuario sea reconocible
    # en la próxima llamada a _load_registered_faces().

    _load_registered_faces.cache_clear()



### FIN DE LO IMPORTANTE

def _preprocesar_para_ocr(imagen):
    """Mejora la imagen para que Tesseract lea mejor."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Aumentar contraste
    gris = cv2.equalizeHist(gris)
    # Umbralización adaptativa — mejor que umbral fijo para documentos
    binaria = cv2.adaptiveThreshold(
        gris, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return binaria


def _extraer_texto(imagen):
    reader = _get_ocr_reader()
    resultados_crudos = reader.readtext(imagen, detail=0, paragraph=True)

    imagen_preprocesada = _preprocesar_para_ocr(imagen)
    resultados_preprocesados = reader.readtext(imagen_preprocesada, detail=0, paragraph=True)

    texto_crudo = "\n".join(resultados_crudos).strip()
    texto_preprocesado = "\n".join(resultados_preprocesados).strip()

    if len(texto_preprocesado) > len(texto_crudo):
        return texto_preprocesado

    return texto_crudo


def _parsear_nombre_cedula(texto: str) -> str | None:
    """
    Busca patrones del carnet colombiano:
        NOMBRES
        JUAN DAVID
        APELLIDOS
        RODRIGUEZ PEREZ
    Devuelve 'juan_david_rodriguez_perez' o None si no encuentra nada.
    """
    import re
    import unicodedata

    def normalizar(texto_base: str) -> str:
        texto_base = unicodedata.normalize("NFD", texto_base)
        texto_base = "".join(c for c in texto_base if unicodedata.category(c) != "Mn")
        texto_base = re.sub(r"[^\w\s:\-]", " ", texto_base, flags=re.UNICODE)
        texto_base = re.sub(r"\s+", " ", texto_base).strip()
        return texto_base

    def limpiar_nombre(valor: str | None) -> str | None:
        if not valor:
            return None

        valor = unicodedata.normalize("NFD", valor)
        valor = "".join(c for c in valor if unicodedata.category(c) != "Mn")
        valor = re.sub(r"[^a-zA-Z\s]", "", valor)
        valor = " ".join(valor.split())
        return valor or None

    texto_normalizado = normalizar(texto)
    patron_nombre = re.compile(
        r"\b(?:primer\s+nombre|segundo\s+nombre|nombres?|name)\b\s*[:\-]?\s*(?P<nombre>[A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]{2,})",
        re.IGNORECASE,
    )
    patron_apellido = re.compile(
        r"\b(?:primer\s+apellido|segundo\s+apellido|apellidos?|surname|last\s*name)\b\s*[:\-]?\s*(?P<apellido>[A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]{2,})",
        re.IGNORECASE,
    )

    nombre_match = patron_nombre.search(texto_normalizado)
    apellido_match = patron_apellido.search(texto_normalizado)

    nombres = limpiar_nombre(nombre_match.group("nombre")) if nombre_match else None
    apellidos = limpiar_nombre(apellido_match.group("apellido")) if apellido_match else None

    if nombres and apellidos:
        nombre_completo = f"{nombres} {apellidos}"
    else:
        return None

    # Limpiar y normalizar: solo letras y espacios → snake_case
    # Quitar tildes
    nombre_completo = unicodedata.normalize("NFD", nombre_completo)
    nombre_completo = "".join(c for c in nombre_completo if unicodedata.category(c) != "Mn")
    # Quitar caracteres que no sean letras ni espacios
    nombre_completo = re.sub(r"[^a-zA-Z\s]", "", nombre_completo)
    # Snake case
    return "_".join(nombre_completo.lower().split())


def extraer_nombre_documento(content: bytes) -> str | None:
    nparr = np.frombuffer(content, np.uint8)
    imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if imagen is None:
        return None

    texto = _extraer_texto(imagen)
    return _parsear_nombre_cedula(texto)


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


def reset_liveness_state():
    liveness_verifier.reset_state()

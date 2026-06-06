import shutil
import time

import cv2
from test_reco import procesar_imagen, refresh_blacklist_cache, verificar_blacklist, verificar_liveliness, verificar_login, reset_liveness_state

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
import uuid
import uvicorn
from pathlib import Path


app = FastAPI()
login_status = {"state": "idle", "message": ""}

BLACKLIST_DIR = Path("blacklist")
BLACKLIST_DIR.mkdir(parents=True, exist_ok=True)
REGISTERED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def set_login_status(state: str, message: str = ""):
    login_status["state"] = state
    login_status["message"] = message


def _encode_mjpeg_frame(frame):
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return None

    return (
        b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        + buffer.tobytes()
        + b'\r\n'
    )


def _put_text(frame, text, position, color, scale=0.7):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def _stream_frame(frame):
    payload = _encode_mjpeg_frame(frame)
    if payload is not None:
        return payload

    return None


def _is_registered_user_folder(path: Path) -> bool:
    return path.is_dir() and path.name not in {"uploads", "models"}


def _find_registered_user_image(user_name: str) -> Path | None:
    user_dir = Path("documentos") / user_name
    if not user_dir.exists():
        return None

    for image_path in sorted(user_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in REGISTERED_EXTENSIONS:
            return image_path

    return None


def _load_registered_users():
    users = []
    documentos_dir = Path("documentos")
    if not documentos_dir.exists():
        return users

    for folder in sorted(documentos_dir.iterdir()):
        if not _is_registered_user_folder(folder):
            continue

        image_path = _find_registered_user_image(folder.name)
        if image_path is None:
            continue

        users.append({"name": folder.name, "image": str(image_path)})

    return users


def _load_blacklisted_users():
    users = []
    for image_path in sorted(BLACKLIST_DIR.glob("*.jpg")):
        users.append({"name": image_path.stem, "image": str(image_path)})

    return users


def _blacklist_user(user_name: str):
    source_image = _find_registered_user_image(user_name)
    if source_image is None:
        raise HTTPException(status_code=404, detail="El usuario no existe en los registrados")

    destination_image = BLACKLIST_DIR / f"{user_name}{source_image.suffix.lower()}"
    shutil.copy2(source_image, destination_image)
    refresh_blacklist_cache()
    return destination_image

@app.get("/")
async def root():
    html_path = Path("views") / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))



@app.get("/register")
async def register_page():
    html_path = Path("views") / "register.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="register.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))



direccion_subidos = Path("documentos") / "uploads"
direccion_subidos.mkdir(parents=True, exist_ok=True)

@app.post("/subir")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".jpg") and not file.filename.endswith(".jpeg") and not file.filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="El archivo no es un formato de imagen válido")
    
    extension = Path(file.filename).suffix.lower()

    nombre_archivo = f"{uuid.uuid4()}{extension}"
    ruta_destino = direccion_subidos / nombre_archivo

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="No se pudo leer el archivo subido")

    # Guardar archivo
    with open(ruta_destino, "wb") as f:
        f.write(content)

    procesar_imagen(file.filename, content)

    await file.seek(0)

    return {
        "filename_original": file.filename,
        "filename_guardado": nombre_archivo,
        "message": "Archivo subido exitosamente"
    }



def generate_frames():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            blacklist_frame, blacklisted, blacklist_label = verificar_blacklist(frame)
            if blacklisted:
                set_login_status("blacklisted", f"Blacklisted user detected: {blacklist_label}")
                _put_text(
                    blacklist_frame,
                    f"BLACKLISTED USER DETECTED: {blacklist_label}",
                    (20, 80),
                    (0, 0, 255),
                    scale=0.9,
                )
                payload = _stream_frame(blacklist_frame)
                if payload is not None:
                    yield payload
                break

            processed_frame, _ = verificar_liveliness(frame)
            payload = _stream_frame(processed_frame)
            if payload is None:
                continue

            yield payload
    finally:
        cap.release()


def _process_liveness_frame(frame, start_time):
    processed_frame, live, status = verificar_liveliness(frame, detail=True)
    elapsed_seconds = time.monotonic() - start_time
    remaining_seconds = max(0.0, 20.0 - elapsed_seconds)
    face_detected = status.get("face_detected", False)
    blink_count = status.get("blink_count", 0)
    left_blink = status.get("left_blink", 0.0)
    right_blink = status.get("right_blink", 0.0)
    average_blink = status.get("average_blink", 0.0)
    ear = status.get("ear", 0.0)
    eye_count = status.get("eye_count", 0)
    source = status.get("source", "mediapipe")
    live_state = "LIVE" if live else "WAITING FOR BLINK"

    _put_text(
        processed_frame,
        f"SOURCE: {source.upper()}",
        (20, 80),
        (255, 255, 0),
    )
    _put_text(
        processed_frame,
        "LIVENESS TEST: blink",
        (20, 110),
        (255, 255, 0),
    )
    _put_text(
        processed_frame,
        f"FACE DETECTED: {'YES' if face_detected else 'NO'}",
        (20, 140),
        (0, 255, 0) if face_detected else (0, 0, 255),
    )
    _put_text(
        processed_frame,
        f"BLINKS DETECTED: {blink_count}",
        (20, 170),
        (0, 255, 0) if blink_count > 0 else (255, 255, 0),
    )
    _put_text(
        processed_frame,
        f"LIVENESS STATE: {live_state}",
        (20, 200),
        (0, 255, 0) if live else (255, 255, 0),
    )
    _put_text(
        processed_frame,
        f"BLENDSHAPE L/R/AVG: {left_blink:.2f}/{right_blink:.2f}/{average_blink:.2f}",
        (20, 230),
        (255, 255, 0),
    )
    _put_text(
        processed_frame,
        f"EAR: {ear:.3f} | EYES: {eye_count}",
        (20, 260),
        (255, 255, 0),
    )
    _put_text(
        processed_frame,
        f"Time left: {remaining_seconds:.1f}s",
        (20, 290),
        (255, 255, 0),
    )

    if live:
        set_login_status("passed", "Liveness test passed")
        _put_text(
            processed_frame,
            "LIVENESS PASSED - checking identity",
            (20, 320),
            (0, 255, 0),
        )
        return processed_frame, "passed"

    if elapsed_seconds >= 20.0:
        set_login_status("failed", "Liveness test failed. Video feed closed.")
        _put_text(
            processed_frame,
            "LIVENESS FAILED - login closed",
            (20, 320),
            (0, 0, 255),
        )
        return processed_frame, "failed"

    return processed_frame, "waiting"


def _process_identity_frame(frame):
    processed_frame, matched, label = verificar_login(frame)

    if matched:
        set_login_status("matched", f"Access granted: {label}")
        _put_text(
            processed_frame,
            f"FACE MATCHED: {label}",
            (20, 80),
            (0, 255, 0),
            scale=0.9,
        )
        _put_text(
            processed_frame,
            "BLINK GATE PASSED - LOGIN READY",
            (20, 115),
            (0, 255, 0),
            scale=0.8,
        )
        return processed_frame, True

    set_login_status("checking_identity", "Checking identity")
    _put_text(
        processed_frame,
        f"FACE NOT RECOGNIZED: {label}",
        (20, 80),
        (0, 0, 255),
        scale=0.9,
    )
    _put_text(
        processed_frame,
        "MATCHING REGISTERED FACES AFTER BLINK TEST",
        (20, 115),
        (255, 255, 0),
        scale=0.8,
    )
    return processed_frame, False


def generate_login_frames():

    reset_liveness_state()
    cap = cv2.VideoCapture(0)
    start_time = time.monotonic()
    liveness_passed = False
    identity_verified = False
    set_login_status("running", "Liveness test running")
    
    frame_count = 0  

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # blacklist cada 15 frames
            if frame_count % 15 == 0:
                blacklist_frame, blacklisted, blacklist_label = verificar_blacklist(frame)
                if blacklisted:
                    set_login_status("blacklisted", f"Blacklisted user detected: {blacklist_label}")
                    _put_text(blacklist_frame, f"BLACKLISTED USER DETECTED: {blacklist_label}", (20, 80), (0, 0, 255), scale=0.9)
                    _put_text(blacklist_frame, "VIDEO FEED CLOSED", (20, 115), (0, 0, 255), scale=0.8)
                    payload = _stream_frame(blacklist_frame)
                    if payload is not None:
                        yield payload
                    break

            elapsed_seconds = time.monotonic() - start_time
            if elapsed_seconds >= 20.0:
                if identity_verified:
                    set_login_status("granted", "Login granted")
                else:
                    set_login_status("failed", "Liveness test failed. Video feed closed.")
                break

            # Liveliness
            if not liveness_passed:
                processed_frame, phase = _process_liveness_frame(frame, start_time)

                if phase == "passed":
                    liveness_passed = True
                    payload = _stream_frame(processed_frame)
                    if payload is not None:
                        yield payload
                    continue
                elif phase == "failed":
                    payload = _stream_frame(processed_frame)
                    if payload is not None:
                        yield payload
                    break
                else:
                    payload = _stream_frame(processed_frame)
                    if payload is not None:
                        yield payload
                    continue

            # face matching cada 5 frames
            if frame_count % 5 == 0:
                processed_frame, matched = _process_identity_frame(frame)
                if matched:
                    identity_verified = True
                    set_login_status("granted", "Login granted")
                    _put_text(processed_frame, "LOGIN VERIFIED - closing at 10 seconds", (20, 115), (0, 255, 0))
            else:
                
                processed_frame = frame.copy()
                _put_text(processed_frame, "LIVENESS PASSED - checking identity", (20, 40), (0, 255, 0))
                if identity_verified:
                    _put_text(processed_frame, "LOGIN VERIFIED", (20, 80), (0, 255, 0), scale=0.9)
                else:
                    _put_text(processed_frame, "MATCHING REGISTERED FACES...", (20, 80), (255, 255, 0), scale=0.9)

            payload = _stream_frame(processed_frame)
            if payload is not None:
                yield payload
    finally:
        cap.release()

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
                            media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/login-video")
def login_video_feed():
    return StreamingResponse(generate_login_frames(),
                            media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/login-status")
async def login_status_view():
    return login_status



@app.get("/blacklist-faces")
async def blacklist_faces():
    return _load_blacklisted_users()

@app.get("/registered-faces")
async def registered_faces():
    return _load_registered_users()


@app.post("/blacklist/{user_name}")
async def add_to_blacklist(user_name: str):
    destination_image = _blacklist_user(user_name)
    return {
        "message": f"{user_name} was added to the blacklist",
        "filename": destination_image.name,
    }

@app.get("/login")
async def login_page():
    html_path = Path("views") / "login.html"
    ...
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/lista-negra")
async def check_blacklist():
    html_path = Path("views") / "blacklist.html"
    ...
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":    
    uvicorn.run(app, host="127.0.0.1", port=8000)

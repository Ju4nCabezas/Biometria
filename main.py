from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn

from backend.app_logic import ( ##Imports del backend
    blacklist_user,
    generate_frames,
    generate_login_frames,
    get_login_status,
    handle_upload_file,
    list_blacklisted_faces,
    list_registered_faces,
    render_html_page,
)


app = FastAPI()

#Este es solo el servidor de FastAPI y los endpoints


@app.get("/")
async def root():
    return render_html_page("index.html")


@app.get("/register")
async def register_page():
    return render_html_page("register.html")


@app.post("/subir")
async def upload_file(file: UploadFile = File(...)):
    return await handle_upload_file(file)


@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/login-video")
def login_video_feed():
    return StreamingResponse(generate_login_frames(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/login-status")
async def login_status_view():
    return get_login_status()


@app.get("/blacklist-faces")
async def blacklist_faces():
    return list_blacklisted_faces()


@app.get("/registered-faces")
async def registered_faces():
    return list_registered_faces()


@app.post("/blacklist/{user_name}")
async def add_to_blacklist(user_name: str):
    return blacklist_user(user_name)


@app.get("/login")
async def login_page():
    return render_html_page("login.html")


@app.get("/lista-negra")
async def check_blacklist():
    return render_html_page("blacklist.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
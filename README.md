# Biometria

Sistema de control de acceso con reconocimiento facial, prueba de liveliness (detección de parpadeo) y lista negra de usuarios bloqueados. Construido con FastAPI, OpenCV, `face_recognition` y MediaPipe.

## Características

- **Login facial en tiempo real** vía stream de cámara con verificación de identidad.
- **Prueba de liveliness** basada en detección de parpadeo (MediaPipe FaceLandmarker, con fallback a Haar Cascades si el modelo no está disponible).
- **Registro de usuarios** mediante carga de imagen, con detección automática de rostro.
- **Extracción de nombre por OCR** desde documentos de identidad (EasyOCR), con reconocimiento de re-registro para evitar duplicados.
- **Lista negra**: bloquea usuarios registrados y los detecta automáticamente durante el login, cortando el stream si aparecen.
- **Interfaz web** con páginas dedicadas para login, registro y administración de blacklist.

## Estructura del proyecto

```
proyecto/
├── main.py                      
├── backend/
│   ├── app_logic.py                
│   ├── reco_logic.py               
│   └── LivenessVerifier.py     
├── views/                          
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   └── blacklist.html
├── documentos/                  
│   ├── <usuario>/                    
│   ├── models/
│   │   └── face_landmarker.task        
│   └── uploads/                       
├── .gitignore
├── requirements.txt
└── test_reco.py                  
```

## Requisitos

- Python 3.10+
- Cámara web disponible en el dispositivo `0`
- EasyOCR

### Dependencias principales

```
fastapi
uvicorn
opencv-python
face_recognition
mediapipe
easyocr
numpy
python-multipart
```

Instalación:

```bash
pip install -r requirements.txt
```

> `face_recognition` requiere `dlib`, que a su vez necesita CMake y un compilador C++. En Linux: `apt install cmake build-essential`. En Windows se recomienda instalar mediante un wheel precompilado.

### Modelo de MediaPipe

Descarga `face_landmarker.task` y colócalo en:

```
documentos/models/face_landmarker.task
```

Si el modelo no está presente, el sistema usa automáticamente un fallback con Haar Cascades para la detección de parpadeo (menos preciso).

## Uso

### Arrancar el servidor

```bash
python main.py
```

El servidor queda disponible en `http://127.0.0.1:8000`.

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Página de bienvenida |
| `GET` | `/login` | Página de login con stream de cámara |
| `GET` | `/login-video` | Stream MJPEG del proceso de login |
| `GET` | `/login-status` | Estado actual del login (polling) |
| `GET` | `/register` | Página de registro de usuario |
| `POST` | `/subir` | Sube una imagen y registra al usuario |
| `GET` | `/registered-faces` | Lista de usuarios registrados |
| `GET` | `/lista-negra` | Página de administración de blacklist |
| `GET` | `/blacklist-faces` | Lista de usuarios bloqueados |
| `POST` | `/blacklist/{user_name}` | Mueve un usuario a la lista negra |
| `GET` | `/video` | Stream MJPEG genérico con chequeo de blacklist |

### Flujo de login

1. El usuario inicia la prueba desde `/login`.
2. Se evalúa la blacklist periódicamente durante el stream; si hay coincidencia, el feed se cierra de inmediato.
3. Se ejecuta la prueba de liveliness (parpadeo) durante un máximo de 20 segundos.
4. Tras superar la prueba, se compara el rostro contra los usuarios registrados.
5. Si hay coincidencia, se concede el acceso y se redirige a `/lista-negra`.

### Flujo de registro

1. El usuario sube una foto de su documento de identidad desde `/register`.
2. El sistema detecta el rostro en la imagen con Haar Cascade.
3. Se extrae el nombre del documento mediante OCR (EasyOCR), buscando etiquetas como `NOMBRES` / `APELLIDOS`.
4. Si el rostro ya coincide con un usuario existente, se reutiliza su nombre (evita duplicados); si no, se usa el nombre extraído por OCR.
5. La imagen del rostro recortado se guarda en `documentos/<nombre_usuario>/`.

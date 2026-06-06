

class LivenessVerifier:
    def __init__(self):
        self.face_landmarker = _load_face_landmarker()
        self.frame_timestamp_ms = 0
        self.closed_eye_frames = 0
        self.blink_count = 0

    def reset_state(self):
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

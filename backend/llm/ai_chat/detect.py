import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(
        model_selection=0,  # 0 = proche, 1 = loin
        min_detection_confidence=0.5) as detector:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe travaille en RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        face_detected = False

        if result.detections:
            face_detected = True

            for detection in result.detections:
                mp_draw.draw_detection(frame, detection)

        print("Face:", face_detected)

        cv2.imshow("MediaPipe Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
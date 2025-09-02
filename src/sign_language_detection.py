import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

# Load the trained model
MODEL_PATH = "model/sign_language_model.h5"
print(f"✅ Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Create a folder for saving videos
video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)

# Get user choice
mode = input("Enter mode (numbers/alphabet): ").strip().lower()
if mode == "numbers":
    classes = list(map(str, range(1, 10)))
elif mode == "alphabet":
    classes = [chr(i) for i in range(65, 91)]
else:
    print("❌ Invalid mode! Defaulting to all signs.")
    classes = list(map(str, range(1, 10))) + [chr(i) for i in range(65, 91)]

# Sentence and cooldown setup
sentence = ""
prev_prediction = None
cooldown_frames = 30  # Number of frames to wait between predictions
frame_counter = 0

# Camera setup
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
video_filename = os.path.join(video_folder, f"captured_signs_{mode}.avi")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(video_filename, fourcc, 10, (frame_width, frame_height))

print("✅ Camera initialized. Press 'q' to quit, 'r' to reset sentence, spacebar to add space.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            margin = 20
            x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
            x_max, y_max = min(x_max + margin, w), min(y_max + margin, h)
            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size != 0:
                hand_crop = cv2.resize(hand_crop, (48, 48))
                hand_crop = hand_crop.astype(np.float32) / 255.0
                hand_crop = np.expand_dims(hand_crop, axis=0)

                if frame_counter >= cooldown_frames:
                    predictions = model.predict(hand_crop, verbose=0)
                    predicted_index = np.argmax(predictions)
                    confidence = predictions[0][predicted_index]

                    if 0 <= predicted_index < len(classes):
                        predicted_label = classes[predicted_index]
                        if confidence > 0.5 and predicted_label != prev_prediction:
                            sentence += predicted_label
                            print(f"✅ Added '{predicted_label}' | Sentence: {sentence}")
                            prev_prediction = predicted_label
                            frame_counter = 0
                    else:
                        print("⚠️ Predicted index out of range.")
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display sentence
    cv2.putText(frame, f"Sentence: {sentence}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Write video and show frame
    out.write(frame)
    cv2.imshow("Sign Language Detection", frame)

    frame_counter += 1
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        sentence = ""
        prev_prediction = None
        print("🔄 Sentence Reset")
    elif key == ord(' '):
        sentence += " "
        print("➕ Space added")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Final Sentence: {sentence}")
print(f"🎥 Video saved at: {video_filename}")

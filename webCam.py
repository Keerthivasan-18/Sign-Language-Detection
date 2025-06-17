import cv2
import mediapipe as mp
import tensorflow as tf
from model import decode
import numpy as np

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load trained model
model = tf.keras.models.load_model('asl_model.h5')

# Font setup
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 0, 255)
thickness = 2

# Camera setup
cap = cv2.VideoCapture(0)

# Variables for buffering predictions
last_char = ''
buffer = ''
words = []

# Output video writer (optional)
out = cv2.VideoWriter('output.mp4', -1, 20.0, (640, 480))

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = image.shape

            xList = [int(lm.x * w) for lm in hand_landmarks.landmark]
            yList = [int(lm.y * h) for lm in hand_landmarks.landmark]
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            roi_x = max(0, xmin - 50)
            roi_y = max(0, ymin - 50)
            roi_w = min(w - roi_x, xmax - xmin + 100)
            roi_h = min(h - roi_y, ymax - ymin + 100)

            roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            try:
                resized = cv2.resize(roi, (128, 128))
                img_array = np.array([resized])
                prediction = model.predict(img_array, verbose=0)
                pred_class = decode(np.argmax(prediction))
                
                # Accumulate characters if prediction is stable
                if pred_class != last_char:
                    if pred_class != 'nothing':
                        buffer += pred_class
                        last_char = pred_class
                    else:
                        if last_char != 'nothing' and buffer:
                            words.append(buffer)
                            buffer = ''
                            last_char = 'nothing'

                # Draw ROI box
                cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

            except Exception as e:
                print("Error:", e)

        # Show prediction as sentence
        sentence = ' '.join(words) + ' ' + buffer
        cv2.putText(image, sentence.strip(), (10, 40), font, font_scale, color, thickness, cv2.LINE_AA)

        out.write(image)
        cv2.imshow('ASL to Text', image)

        # Press Esc to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

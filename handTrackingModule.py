# Creator: Bret Lotti
# Date: 08/17/2023
# Purpose: This python module will track hand movements from a webcam using the OpenCV and mediapipe libraries.

import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # change image to RGB as mediapipe cannot process BGR images
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist
    
def main():
    cap = cv2.VideoCapture(0)
    detector = handTracker()

    # Load the pre-trained gesture recognition model
    gesture_recognizer = load_model('gesture_recognition_model.h5')

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            hand_landmarks = lmlist
            x_sum = 0
            y_sum = 0
            for lm in hand_landmarks:
                x_sum += lm[1]
                y_sum += lm[2]
            hand_center_x = x_sum // len(hand_landmarks)
            hand_center_y = y_sum // len(hand_landmarks)
            
            roi_size = 150  # Adjust this value to control the size of the ROI
            roi = img[hand_center_y - roi_size // 2:hand_center_y + roi_size // 2,
                      hand_center_x - roi_size // 2:hand_center_x + roi_size // 2]
            
            roi = cv2.resize(roi, (195, 240))
            roi = roi / 255.0
            prediction = gesture_recognizer.predict(np.array([roi]))

            gesture_classes = ["call_me", "fingers_crossed", "okay", "paper", "peace", "rock", "rock_on", "scissor", "thumbs", "up"]
            predicted_gesture = gesture_classes[np.argmax(prediction)]

            cv2.putText(img, predicted_gesture, (hand_center_x, hand_center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# def main():
#     cap = cv2.VideoCapture(0)
#     detector = handTracker()

#     while True:
#         success, img = cap.read()
#         img = detector.findHands(img)
#         lmlist = detector.findPosition(img)
#         if len(lmlist) != 0:
#             print(lmlist[4])

#         cv2.imshow("Image", img)
#         key = cv2.waitKey(1)

#         if key & 0xFF == ord('q'):
#             break

# if __name__ == "__main__":
#     main()
        


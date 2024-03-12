import cv2
import numpy as np
import mediapipe as mp

class HandGestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def get_gesture(self, image, debug=False):
        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate the direction the index finger is pointing
                wrist = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * image_width,
                                  hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * image_height])
                index_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                      hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height])

                # Calculate direction vector
                direction = index_tip - wrist

                # Normalize the direction vector
                norm = np.linalg.norm(direction)
                if norm != 0:
                    direction = direction / norm

                # Determine pointing left or right based on the direction vector
                if direction[0] > 0.5:  # Adjust the threshold based on testing
                    gesture = 'Pointing Right'
                elif direction[0] < -0.5:  # Adjust the threshold based on testing
                    gesture = 'Pointing Left'

        if debug:
            cv2.putText(image, gesture if gesture else '', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Return the image with gesture text for debug, and the gesture
        return image, gesture if gesture else ''

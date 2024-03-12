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
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate the direction the index finger is pointing
                wrist_coords = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * image_width,
                                         hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * image_height])
                index_tip_coords = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                             hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height])
                
                # Calculate direction vector from wrist to index tip
                direction = index_tip_coords - wrist_coords
                norm = np.linalg.norm(direction)
                if norm != 0:
                    direction = direction / norm  # Normalize the direction vector

                # Define thresholds for determining if pointing left or right
                if direction[0] > 0.5:  # Adjust the threshold based on your needs
                    pointing_direction = 'Right'
                elif direction[0] < -0.5:  # Adjust the threshold based on your needs
                    pointing_direction = 'Left'
                else:
                    pointing_direction = None

                # Check the relative position of the fingertips to the wrist
                fingertips = [
                    self.mp_hands.HandLandmark.THUMB_TIP,
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    self.mp_hands.HandLandmark.RING_FINGER_TIP,
                    self.mp_hands.HandLandmark.PINKY_TIP
                ]

                fingers_up = []
                for fingertip in fingertips:
                    finger_tip_coords = np.array([hand_landmarks.landmark[fingertip].x * image_width,
                                                  hand_landmarks.landmark[fingertip].y * image_height])
                    # A finger is considered "up" if the tip is above the wrist in the image
                    fingers_up.append(finger_tip_coords[1] > wrist_coords[1])
                
                # Determine if hand is open, closed, or pointing
                if all(fingers_up):
                    gesture = 'Open Palm'
                elif not any(fingers_up):
                    gesture = 'Closed Palm'
                elif pointing_direction and fingers_up[1]:  # Index finger is pointing
                    gesture = f'Pointing {pointing_direction}'
                else:
                    gesture = 'Unknown'

        # if debug and gesture:
        #     cv2.putText(image, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Return the image with gesture text for debug, and the gesture
        return image, gesture

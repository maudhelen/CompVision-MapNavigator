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
                    
                ### Wheter the hand is open or closed
                # Calculate the center of the wrist (base reference for closed hand and dragging)
                wrist = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * image_width,
                                  hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * image_height])

                # Calculate distances between wrist and finger tips (you could also calculate MCP for better accuracy)
                distances = []
                for tip in [self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                            self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.PINKY_TIP]:
                    tip_pos = np.array([hand_landmarks.landmark[tip].x * image_width,
                                        hand_landmarks.landmark[tip].y * image_height])
                    distances.append(np.linalg.norm(tip_pos - wrist))

                # Check if the hand is closed
                if all(dist < 100 for dist in distances):  # You may need to adjust this threshold based on your camera setup
                    if not self.is_closed:
                        self.is_closed = True
                        self.previous_wrist = wrist
                        gesture = 'Closed Palm'
                    elif self.is_closed and self.previous_wrist is not None:
                        # Check if there's significant movement
                        if np.linalg.norm(wrist - self.previous_wrist) > 20:  # Threshold for dragging movement
                            gesture = 'Dragging'
                            self.previous_wrist = wrist
                    else:
                        self.is_closed = False
                        gesture = None  # Not dragging anymore
                else:
                    self.is_closed = False
                    self.previous_wrist = None
                    gesture = None
        
        # Return the image with gesture text for debug, and the gesture
        return image, gesture if gesture else 'Point left to zoom out, point right to zoom in\nClose your hand to drag'

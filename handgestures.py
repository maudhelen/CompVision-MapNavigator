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
        self.previous_gesture = None
        self.start_point = (0, 0)
        self.is_dragging = False

    def get_gesture(self, image, debug=False):
        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Check if the hand is closed
                thumb_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                                      hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * image_height])
                index_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                      hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height])
                distance = np.linalg.norm(thumb_tip - index_tip)

                if distance < 40:  # Threshold for closed hand (you may need to adjust this based on testing)
                    gesture = 'Closed Hand'
                    if self.is_dragging:
                        # Calculate dragging direction based on movement of the center of the hand
                        palm_center = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * image_width,
                                                hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * image_height])
                        if self.previous_gesture == 'Closed Hand':
                            dx, dy = palm_center[0] - self.start_point[0], palm_center[1] - self.start_point[1]
                            if abs(dx) > abs(dy):  # Horizontal movement
                                gesture = 'Drag Right' if dx > 0 else 'Drag Left'
                            else:  # Vertical movement
                                gesture = 'Drag Down' if dy > 0 else 'Drag Up'
                        self.start_point = palm_center
                    else:
                        self.start_point = (thumb_tip[0], thumb_tip[1])
                        self.is_dragging = True
                else:
                    self.is_dragging = False  # Reset dragging status when hand opens

        self.previous_gesture = gesture
        if debug:
            cv2.putText(image, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return gesture

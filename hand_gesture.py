import numpy as np
import mediapipe as mp
import cv2
import math

""" 
This class decides the hand gestures based on the landmarks of the hand and the logic we decide to use.
For a better udnerstanding of the logic you can see the annotated image sin the images folder.
Since we decide the hand gestures are more intuitive without flipping the image, we have an opposite logic in most of the methods
"""

class HandGesture:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    @staticmethod
    def is_palm_open(hand_landmarks):
        #check if the tip of all fingers is above the joint of all fingers (except thumb)
        for tip, joint in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if hand_landmarks[tip].y > hand_landmarks[joint].y:
                return False
        return True
    
    @staticmethod
    # Make sure is_palm_closed is an instance method, not a static method.
    def is_palm_closed(hand_landmarks):
        for tip, joint in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if hand_landmarks[tip].y < hand_landmarks[joint].y: #if the tip of all fingers is below the joint of all fingers (except thumb), then the palm is closed
                return False
        return True
    
    @staticmethod
    def is_pointing(hand_landmarks):
        # Determine if the hand is pointing by checking the index finger's position relative to other fingers
        if hand_landmarks[8].y < hand_landmarks[6].y and \
        hand_landmarks[12].y > hand_landmarks[10].y and \
        hand_landmarks[16].y > hand_landmarks[14].y and \
        hand_landmarks[20].y > hand_landmarks[18].y:
            # Hand is pointing - determine direction
            # Calculate angle of index finger with respect to horizontal
            # Note: Inverting y_diff since y-coordinates increase downwards in image coordinates
            x_diff = hand_landmarks[8].x - hand_landmarks[5].x
            y_diff = hand_landmarks[5].y - hand_landmarks[8].y  # Inverted
            angle = math.atan2(y_diff, x_diff) * (180.0 / math.pi)

            # Determine direction based on angle
            if -45 <= angle <= 45:
                return True, 'Left'
            elif 135 <= angle or angle <= -135:
                return True, 'Right'
            elif 45 < angle < 135:
                return True, 'Up' 
        
        # Hand is not pointing
        return False, None

    
    @staticmethod
    def is_dragging(hand_landmarks, previous_hand_landmarks):
        # If there's no previous data, assume not dragging
        if not previous_hand_landmarks:
            return False

        # Consider dragging if the hand was already closed and has moved significantly
        closed_now = HandGesture.is_palm_closed(hand_landmarks)
        closed_before = HandGesture.is_palm_closed(previous_hand_landmarks)

        if closed_now and closed_before:
            # Calculate movement; for simplicity, we can use the wrist points
            current_wrist = hand_landmarks[0]  # Wrist point is always 0
            previous_wrist = previous_hand_landmarks[0]
            
            dx = current_wrist.x - previous_wrist.x
            dy = current_wrist.y - previous_wrist.y
            
            # Determine if there has been significant movement
            movement_threshold = 0.008  # Adjust based on sensitivity required
            if abs(dx) > movement_threshold or abs(dy) > movement_threshold:
                return True
    
        return False
    
    @staticmethod
    def get_position(hand_landmarks):
        # We'll average the coordinates of all landmarks to find the center
        x_coords = [landmark.x for landmark in hand_landmarks]
        y_coords = [landmark.y for landmark in hand_landmarks]
        avg_x = sum(x_coords) / len(hand_landmarks)
        avg_y = sum(y_coords) / len(hand_landmarks)
            
        # Returning as a tuple
        return avg_x, avg_y
        
        
    def process_image(self, image):
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results.multi_hand_landmarks

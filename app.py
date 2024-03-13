import cv2
import pyautogui
import numpy as np
from hand_gesture import HandGesture

# Initialize the hand gesture recognition
hand_gesture = HandGesture()
cap = cv2.VideoCapture(0)

last_position = None
dragging = False
previous_hand_landmarks = None

while True:
    ret, image = cap.read()
    if not ret:
        break
    
    image = cv2.flip(image, 1)
    hand_landmarks_list = hand_gesture.process_image(image)

    gesture = ""  # Initialize gesture text as empty
    if hand_landmarks_list:
        for handLms in hand_landmarks_list:
            landmarks = [landmark for landmark in handLms.landmark]
            current_position = hand_gesture.get_position(landmarks)
            screen_width, screen_height = pyautogui.size()
            x, y = pyautogui.position()

            if previous_hand_landmarks:
                dragging = hand_gesture.is_dragging(landmarks, previous_hand_landmarks)

            # Map hand coordinates to screen (adjust scaling as needed)
            screen_x = np.interp(current_position[0], [0, 1], [screen_width, 0])
            screen_y = np.interp(current_position[1], [0, 1], [0, screen_height])


            if not gesture:  # Only check for other gestures if no current gesture detected
                closed_palm = hand_gesture.is_palm_closed(landmarks)
                open_palm = hand_gesture.is_palm_open(landmarks)
                point, direction = hand_gesture.is_pointing(landmarks)


                if dragging:
                    gesture = "Dragging"
                    pyautogui.mouseDown(screen_x, screen_y, button='left')
                    pyautogui.moveTo(screen_x, screen_y)
                elif closed_palm:
                    gesture = "Closed Palm"
                    pyautogui.mouseDown(screen_x, screen_y, button='left')
                if open_palm:
                    gesture = "Open Palm"
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    pyautogui.mouseUp()
                    pyautogui.moveTo(screen_x, screen_y)
                elif point:
                    gesture = f"Pointing {direction}"
                    if direction == "Up": #Dont do anything
                        pyautogui.mouseUp(button='left')
                    elif direction == "Left": #scroll down
                        pyautogui.scroll(-1)
                    elif direction == "Right": #scroll up
                        pyautogui.scroll(1)
                        
            else:
                gesture = "Unknown gesture"

            previous_hand_landmarks = landmarks

            if gesture:  # This checks if the gesture string is not empty
                cv2.putText(image, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Escape key to break
        break

cap.release()
cv2.destroyAllWindows()

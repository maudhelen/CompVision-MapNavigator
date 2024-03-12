import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands()

# Function to get thumb orientation
def get_thumb_orientation(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    print(f'Middle: {middle_tip}, Index: {index_tip}')

    if thumb_tip.x < index_tip.x:
        return "Left"
    elif thumb_tip.x > index_tip.x and thumb_tip.y < index_tip.y:
        return "Right"
    elif thumb_tip.y > index_tip.y:
        return "Up"
    elif middle_tip.y-1 <= index_tip.y <= middle_tip.y+1:
        return "Down"
    else:
        return ""

webbrowser.open("https://www.google.com/maps/search/")
pyautogui.moveTo(1200,1200)
# Open webcam
cap = cv2.VideoCapture(0)
first_click = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            try:
                # mp.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                # Get thumb orientation
                thumb_orientation = get_thumb_orientation(hand_landmarks)

                # Initialize click
                if first_click:
                    pyautogui.click()
                    first_click = False

                if thumb_orientation == "Up":
                    pyautogui.press('up')
                elif thumb_orientation == "Left":
                    pyautogui.press('left')
                elif thumb_orientation == "Right":
                    pyautogui.press('right')

            except Exception as ex:
                print(ex)   

            # Display thumb orientation
            cv2.putText(frame, f"Thumb: {thumb_orientation}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Thumb Orientation Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(10)==27:
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

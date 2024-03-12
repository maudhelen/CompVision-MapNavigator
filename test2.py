import cv2
import mediapipe as mp
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

    # print(f'WRIST x: {hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x}, y: {hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y}')
    # print(f'INDEX x: {hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z}, y: {hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y}')
    # print(len(hand_landmarks.landmark))

    # points = np.asarray([hand_landmarks.landmark[0], hand_landmarks.landmark[5], hand_landmarks.landmark[17]])

    # normal_vector = np.cross(points[2] - points[0], points[1] - points[2])

    # normal_vector /= np.linalg.norm(normal_vector)

    # print(normal_vector)

    if thumb_tip.x < index_tip.x:
        return "Left"
    elif thumb_tip.x > index_tip.x and thumb_tip.y < index_tip.y:
        return "Right"
    elif thumb_tip.y > index_tip.y:
        return "Up"
    else:
        return "Down"

# Open webcam
cap = cv2.VideoCapture(0)

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

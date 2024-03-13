import cv2
from hand_gesture import HandGesture

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

            if previous_hand_landmarks:
                dragging = hand_gesture.is_dragging(landmarks, previous_hand_landmarks)

            if dragging:
                gesture = "Dragging"

            if not gesture:  # Only check for other gestures if no current gesture detected
                closed_palm = hand_gesture.is_palm_closed(landmarks)
                open_palm = hand_gesture.is_palm_open(landmarks)
                point_left = hand_gesture.pointing_left(landmarks)
                point_right = hand_gesture.pointing_right(landmarks)

                if closed_palm:
                    gesture = "Closed Palm"
                elif open_palm:
                    gesture = "Open Palm"
                elif point_left:
                    gesture = "Pointing Left"
                elif point_right:
                    gesture = "Pointing Right"
                    
            previous_hand_landmarks = landmarks
            
            if gesture:  # This checks if the gesture string is not empty
                cv2.putText(image, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Escape key to break
        break

cap.release()
cv2.destroyAllWindows()

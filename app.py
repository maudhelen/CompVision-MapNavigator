import cv2
from handgestures import HandGestureRecognition
import pyautogui

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    hand_recognizer = HandGestureRecognition()  # Initialize the hand gesture recognizer

    while True:
        success, image = cap.read()
        if not success:
            continue

        gestures = hand_recognizer.get_gesture(image)

        for gesture in gestures:
            if gesture == 'Zoom In':
                pyautogui.hotkey('ctrl', '+')  # Simulate pressing Ctrl and '+' to zoom in
            elif gesture == 'Zoom Out':
                pyautogui.hotkey('ctrl', '-')  # Simulate pressing Ctrl and '-' to zoom out
            elif gesture == 'Navigate Up':
                pyautogui.press('up')
            elif gesture == 'Navigate Down':
                pyautogui.press('down')
            # Add more actions based on other gestures

        # Display the image and detected gestures
        cv2.putText(image, 'Gestures: ' + ', '.join(gestures), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Hand Gesture Navigation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

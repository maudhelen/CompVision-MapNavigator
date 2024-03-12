import cv2
from handgestures import HandGestureRecognition
import pyautogui
import webbrowser
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=int, default=0.5)

    return parser.parse_args()

def main():
    args = get_args()

    # Initialize camera
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Open Google Maps in default browser
    webbrowser.open('https://www.google.com/maps/@?api=1&map_action=map&basemap=satellite')

    hand_recognizer = HandGestureRecognition()
    dragging = False

    while True:
        ret, image = cap.read()
        if not ret:
            continue

        # Get the gesture from hand recognizer
        _, gesture = hand_recognizer.get_gesture(image, debug=True)

        # Implement gesture actions
        if gesture == 'Pointing Right':
            pyautogui.scroll(100)  # Scroll up to zoom in
        elif gesture == 'Pointing Left':
            pyautogui.scroll(-100)  # Scroll down to zoom out
        elif gesture == 'Closed Palm':
            if not dragging:
                pyautogui.mouseDown()  # Start dragging
                dragging = True
        elif gesture == 'Dragging' and dragging:
            # You can add the logic to move the mouse based on the detected hand position
            pass
        else:
            if dragging:
                pyautogui.mouseUp()  # Stop dragging
                dragging = False

        # Display the gesture on the image
        #and transform from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.putText(image, gesture if gesture else '', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Recognition', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

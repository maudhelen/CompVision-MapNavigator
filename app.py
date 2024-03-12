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
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height


    # Camera preparation ###############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    webbrowser.open('https://www.google.com/maps/@?api=1&map_action=map&basemap=satellite', new=1, autoraise=True) 

    cap = cv2.VideoCapture(0)
    hand_recognizer = HandGestureRecognition()

    while True:
        ret, image = cap.read()
        if not ret:
            continue

        # Get the gesture and the image with debug info if needed
        image, gesture = hand_recognizer.get_gesture(image, debug=True)
        print(gesture)
        # Transform the image to the correct color space for PyAutoGUI
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if gesture == 'Pointing Right':
            pyautogui.scroll(1)  # Scroll up to zoom in
        elif gesture == 'Pointing Left':
            pyautogui.scroll(-1)  # Scroll down to zoom out

        # Show the image with debug information
        #add text to show which gesture is being performed
        cv2.putText(image, gesture if gesture else '', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

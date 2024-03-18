import cv2
from keras_helper import HandGestureRecognition
import pyautogui
import webbrowser
import argparse
from model import PointHistoryClassifier, KeyPointClassifier
import csv
import copy
import numpy as np
from collections import deque, Counter


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
    # Get the arguments from the command line
    args = get_args()

    # Initalise and opn the models / classes
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    hgr = HandGestureRecognition()
    
    with open('keras_model/model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    with open(
            'keras_model/model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # Set up the point history and finger gesture history so we can keep track of movements
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # Initialize camera
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Open Google Maps in default browser
    webbrowser.open('https://www.google.com/maps/@?api=1&map_action=map&basemap=satellite')

    # Initialise certain variables to use in our main code base
    dragging = False
    landmark_list = None
    gesture = None 
    hand_sign_id = None
    finger_gesture_id = None
    most_common_fg_id = None
    classifiedhistorypoint = None

    # Main loop
    while True:
        # Read the frame from the camera
        ret, image = cap.read()
        if not ret:
            break

        # Flip image so it looks normal
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)
        # Detection implementation 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get the results from the hand recognition model in mediapipe
        results = hgr.hands.process(image)
        hand_landmarks = results.multi_hand_landmarks

        # Process each hand detected and turn it into a list of landmarks
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                landmark_list = hgr.calc_landmark_list(debug_image, hand_landmarks)
                debug_image = hgr.draw_landmarks(debug_image, landmark_list)
                pre_processed_landmark_list = hgr.pre_process_landmark(landmark_list)
                pre_processed_point_history_list = hgr.pre_process_point_history(debug_image, point_history)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                # If we have a pointing gesture, add the point to the point history so we can track the movement
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()
                classifiedhistorypoint = point_history_classifier_labels[most_common_fg_id[0][0]]

                # Draw the image and display it
                debug_image = hgr.draw_landmarks(debug_image, landmark_list)

                # After processing gestures and obtaining hand_sign_id and most_common_fg_id
                # Map hand coordinates to screen (adjust scaling as needed)
                center_of_hand = landmark_list[0]
                screen_width, screen_height = pyautogui.size()
                screen_x = np.interp(center_of_hand[0], [0, args.width], [0, screen_width])
                screen_y = np.interp(center_of_hand[1], [0, args.height], [0, screen_height])
                dragging = hgr.handle_gesture_actions(hand_sign_id, screen_x, screen_y, dragging, classifiedhistorypoint)

        else:
            # If no hand is detected, add a 0,0 point to the point history so we can track the movement
            point_history.append([0, 0])

        hand_sign_id_names = ['Open', 'Close', 'Pointer', 'OK']
        most_common_fg_id_names = ['Stop', 'Clockwise', 'Counter Clockwise', 'Move']

        if hand_sign_id is not None:
            print("Hand id and name",hand_sign_id, hand_sign_id_names[hand_sign_id])
            print('point history name and id',classifiedhistorypoint, most_common_fg_id_names[most_common_fg_id[0][0]])

        # Display the gesture on the image
        handsignname = hand_sign_id_names[hand_sign_id] if hand_sign_id is not None else '0'
        finger_gesture_name = classifiedhistorypoint if classifiedhistorypoint is not None else '0'
        bothnames = handsignname + finger_gesture_name
        cv2.putText(debug_image, bothnames if hand_sign_id else '', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Recognition', debug_image)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
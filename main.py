import os
from datetime import datetime
import cv2
import dlib
import pyautogui
from imutils import face_utils
from scipy.spatial import distance


def eye_aspect_ratio(eye):
    first_fifth = distance.euclidean(eye[1], eye[5])
    second_fourth = distance.euclidean(eye[2], eye[4])
    zero_three = distance.euclidean(eye[0], eye[3])
    ratio = (first_fifth + second_fourth) / (2.0 * zero_three)
    return ratio


now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")
print("date and time =", dt_string)
date = dt_string + ".jpg"
print(date)

min_ear = 0.25
max_frame = 80
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(".\shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
frame_count = 0
while True:
    ret, frame = cap.read()
    cv2.namedWindow("VideoCapture", cv2.WINDOW_NORMAL)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
        left_Eye = shape[lStart:lEnd]
        right_Eye = shape[rStart:rEnd]
        left_EAR = eye_aspect_ratio(left_Eye)
        right_EAR = eye_aspect_ratio(right_Eye)
        ear = (left_EAR + right_EAR) / 2.0
        leftEyeHull = cv2.convexHull(left_Eye)
        rightEyeHull = cv2.convexHull(right_Eye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 85), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 85), 1)
        if ear < min_ear:
            frame_count += 1
            print(frame_count)
            if frame_count >= max_frame:
                cv2.putText(frame, "!!!!!!!  ALERT! YOU ARE ABOUT TO SLEEP !!!!!!!", (80, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H%M%S")
                print("date and time =", dt_string)
                date = dt_string + ".jpg"
                print(date)
                myScreenshot = pyautogui.screenshot()
                myScreenshot.save('C:\\Users\\Dell\\OneDrive\\Desktop\\WorkShots\\' + date)
                print("img saved")
                os.system("shutdown -l")
        else:
            frame_count = 0

    cv2.imshow("VideoCapture", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()

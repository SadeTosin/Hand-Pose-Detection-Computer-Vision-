

#########################################################
#!pip install mediapipe

#########################################################
# step 1: Import all necessary libraries
import cv2
import mediapipe as mp


#########################################################
# step 2: Identify webcam
cap = cv2.VideoCapture(0)

#########################################################
# leveraging the mediapipe Library used for Pose detection
mpPose = mp.solutions.pose
pose = mpPose.Pose()
# pose = mpPose.pose(static_image_mode = False, upper body only = False, smooth Landmark =True, min_detection_confidence = 0.5)


#########################################################
# To draw and connect the landmarks
mpDraw = mp.solutions.drawing_utils

#########################################################
# switch on your cam
while True:
    _, img = cap.read()
    
    # Convert video/image from RGB to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply the mediapipe pose detection module for detection
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    
    # Draw Landmarks
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
    cv2.imshow("Pose Detection", img)
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break
    
#########################################################        
# Release the capture once all the processing is done.
cap.release()
cv2.destroyAllWindows()
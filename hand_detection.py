
#########################################################
#pip install mediapipe --user

# Create a virtual environment
#python -m venv myenv

# Activate the virtual environment (Windows)
#myenv\Scripts\activate

# Install mediapipe within the virtual environment
# pip install mediapipe

#########################################################
# import all neccessary libraries
import cv2
import mediapipe as mp

#########################################################
# step 2: Identify all necessary libraries
cap = cv2.VideoCapture(0) 
mpHands= mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 

#########################################################
# step 3: Switch on webcam
while True:
    _, img = cap.read()
    
    #convert image from BG to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Apply mediapipe
    results = hands.process(imgRGB)
    
    # print (results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cv2.putText(img, "Hand Detection Program", (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
    cv2.imshow("Hands Detection Program", img)
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break
        
# Release the capture once all the processing is done.
cap.release()
cv2.destroyAllWindows()

# Import required libraries
import cv2
import numpy as np
import time
from cvzone.FaceMeshModule import FaceMeshDetector  # For face landmark detection
import cvzone  # For easier OpenCV handling and overlaying PNGs
import os
import random

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Initialize face mesh detector with maximum one face
detector = FaceMeshDetector(maxFaces=1)

# Landmark IDs for detecting mouth openness and face center
idList = [0, 17, 78, 292]

# Load all eatable images from 'eatables' folder
folderEatable = "eatables"
listEatable = os.listdir(folderEatable)
eatables = [cv2.imread(f'{folderEatable}/{object}', cv2.IMREAD_UNCHANGED) for object in listEatable]

# Load all non-eatable images from 'noneatable' folder
foldernonEatable = "noneatable"
listNonEatable = os.listdir(foldernonEatable)
noneatables = [cv2.imread(f'{foldernonEatable}/{object}', cv2.IMREAD_UNCHANGED) for object in listNonEatable]

# Initialize the first object as eatable
currentobj = random.choice(eatables)
pos = [300, 0]  # Starting position of object
speed = 5       # Falling speed of the object
counts = 0      # Score counter

# Global game state variables
global iseatable
iseatable = True
global gameover
gameover = False

# Function to reset the falling object (randomly eatable or not)
def resetobjects():
    global currentobj, pos, iseatable
    if random.choice([True, False]):
        currentobj = random.choice(eatables)
        iseatable = True
    else:
        currentobj = random.choice(noneatables)
        iseatable = False
    pos[0] = random.randint(100, 1180)  # Random x-position
    pos[1] = 0  # Reset y-position to top

# Game loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip image horizontally for mirror effect

    if not gameover:
        # Detect face mesh landmarks
        img, faces = detector.findFaceMesh(img, draw=False)

        # Draw the current object on screen
        small = cv2.resize(currentobj, (100, 100))
        img = cvzone.overlayPNG(img, small, pos)
        pos[1] += speed  # Move object down

        # If object reaches bottom, reset it
        if pos[1] > 520:
            resetobjects()

        # If face detected
        if faces:
            face = faces[0]
            up = face[idList[0]]
            down = face[idList[1]]

            # Measure vertical (mouth open) and horizontal (face width) distances
            updown, _ = detector.findDistance(face[idList[0]], face[idList[1]])
            lefright, _ = detector.findDistance(face[idList[2]], face[idList[3]])

            # Center of the mouth
            cx, cy = (up[0] + down[0]) // 2, (up[1] + down[1]) // 2

            # Distance from mouth center to falling object center
            dismmouthline, _ = detector.findDistance((cx, cy), (pos[0] + 50, pos[1] + 50))

            # Ratio to determine if mouth is open
            ratio = (updown / lefright) * 100
            mouthstatus = "open" if ratio > 80 else "close"

            # If object is near the mouth and mouth is open
            if dismmouthline < 100 and ratio > 80:
                if iseatable:
                    resetobjects()
                    counts += 1  # Increase score
                else:
                    gameover = True  # End game on wrong object

        # Display score
        cv2.putText(img, str(counts), (120, 120), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 5)

    else:
        # Game Over screen
        cv2.putText(img, "Game Over", (150, 300), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 5)
        cv2.putText(img, "Press 'R' to Restart", (180, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    # Show final frame
    cv2.imshow('image', img)

    # Keypress handling
    key = cv2.waitKey(1)
    if key == ord('r'):
        resetobjects()
        gameover = False
        counts = 0
        iseatable = True
        currentobj = random.choice(eatables)

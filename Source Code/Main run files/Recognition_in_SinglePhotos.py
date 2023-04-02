from algorithms.wide_resnet import WideResNet
from keras_vggface.utils import preprocess_input
from keras.preprocessing.image import img_to_array
from keras_vggface import VGGFace
from keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.face_utils import FaceAligner
from datetime import date,datetime
from os import listdir
from os.path import isfile, join
import numpy as np
import imutils
import pickle
import time
import cv2
import os
import openpyxl
import requests
import dlib

#Paths for all the required modules
prototxt=r"C:\Users\arpit\OneDrive\Desktop\FACE_RECOG_PROTOYPE\Face_detection_model\deploy.prototxt.txt"
model=r"C:\Users\arpit\OneDrive\Desktop\FACE_RECOG_PROTOYPE\Face_detection_model\res10_300x300_ssd_iter_140000.caffemodel"
F_model = r"C:\Users\arpit\OneDrive\Desktop\FACE_RECOG_PROTOYPE\Face_recognition_model\25_people.model"
F_le = r"C:\Users\arpit\OneDrive\Desktop\FACE_RECOG_PROTOYPE\Face_recognition_model\25_people_le.pickle"
E_model_ = r"C:\Users\arpit\OneDrive\Desktop\FACE_RECOG_PROTOYPE\Emotion_detection_model\fer2013_mini_XCEPTION.119-0.65.hdf5"
AG_weights = r"C:\Users\arpit\OneDrive\Desktop\FACE_RECOG_PROTOYPE\Age_Gender_detection_model\weights.18-4.06.hdf5"
shape_predict = r"C:\Users\arpit\OneDrive\Desktop\FACE_RECOG_PROTOYPE\Align_image_model\shape_predictor_68_face_landmarks.dat"

# load our serialized face detector from disk
print("[INFO] Loading Face Detector ")
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

#Loading Neural network model
print("[INFO] Loading Face Recognizer ")
model = load_model(F_model)

le = pickle.loads(open(F_le, "rb").read())
print(le.classes_)

#Loading emotion model and weights
print("[INFO] Loading Emotion Detector ")
E_model = load_model(E_model_, compile=False)
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

#Loading Age and gender model
print("[INFO] Loading Age&Gender Detector ")
face_size = 64
depth=16
width=8
AG_model = WideResNet(face_size, depth=depth, k=width)()
AG_model.load_weights(AG_weights)

#loading face alligner
print("[INFO] Loading Face Alligner ")
predictor = dlib.shape_predictor(shape_predict)
fa = FaceAligner(predictor, desiredFaceWidth=150)

#Start the FPS throughput estimator
fps = FPS().start()

#Reading the frame
frame = cv2.imread(r"C:\Users\arpit\OneDrive\Desktop\FACE_RECOG_PROTOYPE\Test_images\t10.jpg")
frame = imutils.resize(frame, width=650, height=600)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
bboxes=[]
(h, w) = frame.shape[:2]

imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)

#apply OpenCV's deep learning-based face detector to localize faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    #filter out weak detections
    if confidence > 0.70:
        
        #compute the (x, y)-coordinates of the bounding box for the face
        startX = int(detections[0, 0, i, 3] * w)
        startY = int(detections[0, 0, i, 4] * h)
        endX = int(detections[0, 0, i, 5] * w)
        endY = int(detections[0, 0, i, 6] * h)

        #extract the face ROI
        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        gender_face = face
        gray_face = gray[startY:endY, startX:endX]
        (gH, gW) = gray.shape[:2]

        #ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            continue
        elif gW <20 or gH <20:
            continue

        #NEw AGE AND GENDER MODEL
        AG_face = cv2.resize(face, (64,64))
        AG_face = AG_face.astype("float")
        AG_face = img_to_array(AG_face)
        AG_face = np.expand_dims(AG_face, axis=0)
        results = AG_model.predict(AG_face)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        #appending total rectangles in a list
        rect = (startX, startY, endX, endY)
        bboxes.append(rect)

        #ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            continue
                                
        #extract the face ROI and then preproces it in the exact manner as our training data
        face = face.astype("float")
        face= preprocess_input(face)
        face = cv2.resize(face, (150,150))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        #using face aligner
        #rect = dlib.rectangle(startX, startY, endX, endY)
        #aligned_face = aligned_face.astype("float")
        #aligned_face = preprocess_input(aligned_face)
        #aligned_face = cv2.resize(aligned_face, (150,150))
        #cv2.imshow("aligned face", aligned_face)
        #aligned_face = aligned_face.astype("float") / 255.0
        #aligned_face = img_to_array(aligned_face)
        #aligned_face = np.expand_dims(aligned_face, axis=0)

        #Predicting the class labels using our trained model
        preds = model.predict(face)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        #extract the face ROI for our emotion model
        def process_input(x, v2=True):
            x = x.astype('float32')
            x = x / 255.0
            if v2:
                x = x - 0.5
                x = x * 2.0
            return x
        
        #predicting the emotion of the face in frame
        gray_face = cv2.resize(gray_face, (48,48))
        gray_face = process_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = E_model.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        # draw the bounding box of the face along with the associated probability
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (255, 255, 255), 2)
        cv2.rectangle(frame, (0, h - ((i * 20) + 5)), (700, h - ((i * 20) + 60)),
            (255, 255, 255), cv2.FILLED)

        #Printing the found name on screen only if cofidence > 50%
        if proba>0.5:
            text = " NAME : {}({:.1f}%)".format(name, proba*100)
            cv2.putText(frame, text, (2, h - ((i * 20) + 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)

            text1 = " EMOTION : {} ".format(emotion_text)
            cv2.putText(frame, text1, (210, h - ((i * 20) + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

            for x in range(0, len(predicted_genders)):
                text2 = " GENDER : {} ".format("F" if predicted_genders[x][0]> 0.5 else "M")
                cv2.putText(frame, text2, (480, h - ((i * 20) + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,0), 2)
            
            text3 = " AGE : {} ".format(int(predicted_ages)) 
            cv2.putText(frame, text3, (480, h - ((i * 20) + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,0), 2)
            
            text = " TOTAL IN FRAME : {}".format(len(bboxes))
            cv2.putText(frame, text, (2, h - ((i * 20) + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)

        else:
            text= "Unknown"
            cv2.putText(frame, text, (2, h - ((i * 20) + 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)


#update the FPS counter
fps.update()

#show the output frame
cv2.imshow("Frame", frame)
cv2.imwrite(r"C:\Users\arpit\OneDrive\Desktop\FACE_RECOG_PROTOYPE\Proofs\{}.png".format(name), frame)
key = cv2.waitKey(0)

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()

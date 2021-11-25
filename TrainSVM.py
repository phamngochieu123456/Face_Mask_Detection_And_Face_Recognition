from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
import shutil
import cv2
from os import listdir
import pickle

def detect_face(frame, faceNet):
	(h, w) = frame.shape[:2] 
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(startX, startY) = (min(w-1, startX), min(h-1, startY))
			(endX, endY) = (max(0, endX), max(0, endY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			if startX==endX or startY==endY:
				break

			face = frame[startY:endY, startX:endX].copy()
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (160,160))
			face = img_to_array(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")

	return (locs, faces)

def load_faces(directory):
	faces = []
	for filename in listdir(directory):
		path = directory + "/" + filename
		print(path)
		face = cv2.imread(path)
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (160,160))
		face = img_to_array(face)
		faces.append(face)
	faces = np.array(faces, dtype="float32")
	return faces

def load_dataset(directory):
	X = []
	y = []
	for subdir in listdir(directory):
		path = directory + "/" + subdir
		faces = load_faces(path)
		labels = [subdir for _ in range(len(faces))]
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		X.extend(faces)
		y.extend(labels)
	return np.asarray(X), np.asarray(y)

def get_embedding(model, face_pixels):
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = np.expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

X,y = load_dataset("D:\Git\Face_Mask_Detection_And_Face_Recognition\Face_Rec_DataSet\Train")

keras_model = load_model(r"D:\Git\Face_Mask_Detection_And_Face_Recognition\Models\facenet_keras2")
prototxtPath = r"D:\Git\Face_Mask_Detection_And_Face_Recognition\Models\deploy.prototxt"
weightsPath = r"D:\Git\Face_Mask_Detection_And_Face_Recognition\Models\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

newX = []
for face_pixels in X:
	embedding = get_embedding(keras_model, face_pixels)
	newX.append(embedding)
newX = np.asarray(newX)
print(newX.shape)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(newX, y)

pkl_filename = "D:\Git\Face_Mask_Detection_And_Face_Recognition\Models\svm_model3.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(svm_model, file)
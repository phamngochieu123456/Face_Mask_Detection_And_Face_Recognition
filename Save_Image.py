from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import glob
import random
import shutil

def Save_Image(directory):
	prototxtPath = r"D:\Git\Face_Mask_Detection_And_Face_Recognition\Models\deploy.prototxt"
	weightsPath = r"D:\Git\Face_Mask_Detection_And_Face_Recognition\Models\res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	for filename in os.listdir(directory):
		a = 0
		b = 0
		temp = 0

		if not os.path.isdir("D:\Git\Face_Mask_Detection_And_Face_Recognition\Face_Rec_DataSet\Train"+"\\"+filename[:-4]):
			os.mkdir("D:\Git\Face_Mask_Detection_And_Face_Recognition\Face_Rec_DataSet\Train"+"\\"+filename[:-4])
		if not os.path.isdir("D:\Git\Face_Mask_Detection_And_Face_Recognition\Face_Rec_DataSet\Test"+"\\"+filename[:-4]):
			os.mkdir("D:\Git\Face_Mask_Detection_And_Face_Recognition\Face_Rec_DataSet\Test"+"\\"+filename[:-4])

		path_dir = directory + "\\" + filename
		cap = cv2.VideoCapture(path_dir)
		length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		temp = length//120

		while True:
			success, img = cap.read()
			a = a + 1
			if success:
				if a % temp == 0:
					(h, w) = img.shape[:2]
					blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224),(104.0, 177.0, 123.0))
					faceNet.setInput(blob)
					detections = faceNet.forward()
					for i in range(0, detections.shape[2]):
						confidence = detections[0, 0, i, 2]
						if confidence > 0.5:
							box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
							(startX, startY, endX, endY) = box.astype("int")
							(startX, startY) = (max(0, startX), max(0, startY))
							(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
							face = img[startY:endY, startX:endX]
							cv2.imwrite("D:\Git\Face_Mask_Detection_And_Face_Recognition\Face_Rec_DataSet\Train"+"\\"+filename[:-4]+"\\"+str(b)+".jpg",face)
							b = b + 1  
							break		  
			else:
				break
		cap.release()
		cv2.destroyAllWindows()

		for c in random.sample(glob.glob("D:\Git\Face_Mask_Detection_And_Face_Recognition\Face_Rec_DataSet\Train"+"\\"+filename[:-4]+"\\*"),20):
			shutil.move(c,"D:\Git\Face_Mask_Detection_And_Face_Recognition\Face_Rec_DataSet\Test"+"\\"+filename[:-4])

Save_Image("D:\Git\Face_Mask_Detection_And_Face_Recognition\Face_Rec_Video")
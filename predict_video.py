# USAGE
# python predict_video.py --input example_clips/rouen_video.avi

from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import datetime

# to supress AVX2 FMA warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# putText() renders the text to be displayed onto the output video
def putText(text,passCL,pedestrianCL,yboxCL):
	
	#generic settings
	font_style=cv2.FONT_HERSHEY_DUPLEX
	font_size=0.8
	font_thickness=2
	
	#to print current date and time
	text0=str(datetime.datetime.now())
	position0=(30,50)

	text1="no violation"
	text1a=  "- " + str(round(passCL*100,2)) + "%"
	position1=(30,75)
	position1a=(200,75)
	text2="z.crossing"
	text2a= "- " + str(round(pedestrianCL*100,2)) + "%"
	position2=(30,100)
	position2a=(200,100)
	text3="yellow box"
	text3a="- " + str(round(yboxCL*100,2)) + "%"
	position3=(30,125)
	position3a=(200,125)
	
	text4=">"
	if(str(text)=="pass"):
		rgb_value=(0,255,0)	
		position4=(10,75)
	elif(str(text)=="pedestrian"):
		rgb_value=(0,0,255)
		position4=(10,100)
	elif(str(text)=="ybox"):
		rgb_value=(0,0,255)
		position4=(10,125)
		
	cv2.putText(output,text0,position0,font_style,font_size,rgb_value,font_thickness)
	cv2.putText(output,text1,position1,font_style,font_size,rgb_value,font_thickness)
	cv2.putText(output,text1a,position1a,font_style,font_size,rgb_value,font_thickness)
	cv2.putText(output,text2,position2,font_style,font_size,rgb_value,font_thickness)
	cv2.putText(output,text2a,position2a,font_style,font_size,rgb_value,font_thickness)
	cv2.putText(output,text3,position3,font_style,font_size,rgb_value,font_thickness)
	cv2.putText(output,text3a,position3a,font_style,font_size,rgb_value,font_thickness)
	cv2.putText(output,text4,position4,font_style,font_size,rgb_value,font_thickness)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,help="path to our input video")
args = vars(ap.parse_args())

args["model"]="model/activity.model"
args["label_bin"]="model/lb.pickle"
args["output"]="output/lifting_128avg.avi"
args["size"]=1
# load the trained model and label binarizer from disk
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean

	# make predictions on the frame and then update the predictions
	# queue
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)

	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]

	putText(label,preds[0],preds[1],preds[2])
	print(label)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.cv.CV_FOURCC(*"MJPG")
		writer = cv2.VideoWriter(args["output"],fourcc,30,(W,H),True)

	# write the output frame to disk
	writer.write(output)

	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# release the file pointers
writer.release()
vs.release()
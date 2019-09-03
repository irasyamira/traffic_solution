# USAGE
# python predict_video.py --model model/activity.model --label-bin model/lb.pickle --input example_clips/stmarc_video.avi --output output/lifting_128avg.avi --size 128

# import the necessary packages
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

averaging_on=False

# putText() renders the text to be displayed onto the output video
def putText(violation_code):
	font_style=cv2.FONT_HERSHEY_DUPLEX
	position0=(15,50)
	position1=(15,75)
	position2=(15,100)
	font_size=0.8
	font_thickness=2
	now = datetime.datetime.now()
	text0=str(now)
	text1="traffic light state : red"
	#text2= "violation type : running over red light (" + str(violation_code) + ")"
	text2= str(text)
	rgb_value=(0,0,255) #red when there is violation
	# draw the activity on the output frame
	# text should be the violation code
	if (text2=="v"):
		#text= text + str(violation_code)
		rgb_value=(0,0,255) #red when there is violation
	else:
		#text= text + "none"
		rgb_value=(0,255,0) #green when there is no violation
		
	cv2.putText(output,text0,position0,font_style,font_size,rgb_value,font_thickness)
	cv2.putText(output,text1,position1,font_style,font_size,rgb_value,font_thickness)
	cv2.putText(output,text2,position2,font_style,font_size,rgb_value,font_thickness)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True,help="path to our input video")
ap.add_argument("-o", "--output", required=True,help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
#print("[INFO] loading model and label binarizer...")
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

	text = "violation : {}".format(label)
	#violation_code=2
	putText(text)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		#fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		fourcc = cv2.cv.CV_FOURCC(*"MJPG")
		writer = cv2.VideoWriter(args["output"],fourcc,30,(W,H),True)

	# write the output frame to disk
	writer.write(output)

	# show the output image
	cv2.imshow("Output", output)
	#dir_path = os.path.dirname(os.path.realpath(__file__))
	#open(str(dir_path) + "/output/lifting_128avg.avi","r")
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# release the file pointers
#print("[INFO] cleaning up...")
writer.release()
vs.release()
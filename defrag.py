import cv2
import os
import time

current_dir=os.path.dirname(os.path.realpath(__file__))
input_path=current_dir+'/dataset_crop_split/0/'
output_path = current_dir+'/dataset_crop_split/0_output/'

def defrag(vid_file,i):
	vidcap = cv2.VideoCapture(vid_file)
	success,image = vidcap.read()
	count = 0
	while success:
		cv2.imwrite(output_path+str(i)+'-'+"frame%d.jpg" % count, image) # save frame as JPEG file	  
		success,image = vidcap.read()
		count += 1
		
def main():
	print("defrag. in progress..")
	start = time.time()
	i=0
	for filename in os.listdir(input_path):
		if filename.endswith(".mp4"): 
			i+=1
			defrag(input_path+filename,i)
		else:
			continue
	print('number of files defragmented: ' + str(i))
	end = time.time()
	print('time taken to defrag. dataset: ' + str(end - start))
	
main()


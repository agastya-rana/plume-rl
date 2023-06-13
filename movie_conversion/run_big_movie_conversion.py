import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2
import imageio

movie_path = os.path.join('..','src', 'data', 'plume_movies', 'wider_longer_intermittent_2023-6-8_basic_sub.mp4')
video_capture = cv2.VideoCapture(movie_path)

mask = np.load('longer_wider_intermittent_mask.npy')

video_capture.set(cv2.CAP_PROP_POS_FRAMES, 60)
_, bck_frame = video_capture.read()
bck_frame = bck_frame[:,:,2]

output_file = 'longer_wider_intermittent_full_subtraction.mp4'
frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

writer = imageio.get_writer(output_file, fps=frame_rate)

for i in range(0,3800):

	video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
	_, frame_1 = video_capture.read()
	frame_1 = frame_1[:,:,2]

	frame = frame_1.astype(float) - bck_frame.astype(float)
	frame[frame<0] = 0
	frame = frame*mask
	frame = frame.astype(np.uint8)
	writer.append_data(frame)


writer.close()
	


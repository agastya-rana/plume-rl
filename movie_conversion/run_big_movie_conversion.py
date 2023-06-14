import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2
import imageio

movie_path = os.path.join('..','src', 'data', 'plume_movies', 'longer_wider_intermittent_full_subtraction.mp4')
video_capture = cv2.VideoCapture(movie_path)

#mask = np.load('longer_wider_intermittent_mask.npy')

#video_capture.set(cv2.CAP_PROP_POS_FRAMES, 60)
#_, bck_frame = video_capture.read()
#bck_frame = bck_frame[:,:,2]

output_file = 'longer_wider_intermittent_final.mp4'

frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

writer = imageio.get_writer(output_file, fps=frame_rate)

int_list = np.load('source_int_over_time.npy')

good_ints = int_list[150:]
good_frames = 150 + np.arange(len(good_ints))

m, b = np.polyfit(good_frames, good_ints, 1)

init_val = m*150 + b

for i in range(150,3384):

	scale = init_val/(m*i + b)

	video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
	_, frame_1 = video_capture.read()
	frame_1 = frame_1[:,:,2]

	frame_1 = frame_1.astype(float)
	frame = scale*frame_1
	frame = frame.astype(np.double)
	writer.append_data(frame)


writer.close()
	


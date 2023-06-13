import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2
from matplotlib.colors import LogNorm

movie_path = os.path.join('..','src', 'data', 'plume_movies', 'wider_longer_intermittent_2023-6-8_basic_sub.mp4')
video_capture = cv2.VideoCapture(movie_path)


video_capture.set(cv2.CAP_PROP_POS_FRAMES, 60)
_, frame_1 = video_capture.read()
frame_1 = frame_1[:,:,2].T

plt.imshow(frame_1)
plt.show()


video_capture.set(cv2.CAP_PROP_POS_FRAMES, 400)
_, frame_2 = video_capture.read()
frame_2 = frame_2[:,:,2].T

plt.imshow(frame_2)
plt.show()

sub = frame_2.astype(float) - frame_1.astype(float)
sub[sub<0] = 0

plt.imshow(sub)
plt.show()

num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print('total frames = ', num_frames)
print('fps = ', video_capture.get(cv2.CAP_PROP_FPS))

sum_frame = np.zeros(np.shape(frame_1))

"""

for i in range(0,3800):

	print(i)
	video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
	_, frame_1 = video_capture.read()
	frame_1 = frame_1[:,:,2].T
	sum_frame += frame_1


mean_frame = (sum_frame/3800).astype(float)
mean_frame = mean_frame - frame_1.astype(float)
mean_frame[mean_frame<0] = 0

"""

#np.save('longer_wider_intermittent_mean_frame.npy', mean_frame)

"""
mean_frame = np.load('longer_wider_intermittent_mean_frame.npy')
#mean_frame[mean_frame==0]=np.nan
mean_frame = mean_frame/np.max(mean_frame)
print(np.shape(mean_frame))


x_coords = 0.154*np.arange(0,np.shape(mean_frame)[1])
x_coords = x_coords - 600*0.154
x_arr = np.tile(x_coords, (np.shape(mean_frame)[0], 1))

plt.imshow(x_arr)
plt.show()

y_coords = 0.154*np.arange(0,np.shape(mean_frame)[0])-80*0.154
print(y_coords[-1])

y_arr = (np.tile(y_coords, (1208,1))).T
print(np.shape(y_arr))

plt.imshow(y_arr)
plt.show()

factor = y_coords[-1]/74.5**2

mask = (y_arr > factor*x_coords**2)

np.save('longer_wider_intermittent_mask.npy', mask.T)

plt.imshow(mean_frame.T, norm=LogNorm(vmin=0.001, vmax=1))

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('intensity')
plt.show()

plt.imshow(mean_frame.T*mask.T, norm=LogNorm(vmin=0.001, vmax=1))
# Add colorbar
cbar = plt.colorbar()
cbar.set_label('intensity')
plt.show()

"""

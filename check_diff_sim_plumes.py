import numpy as np 
from src.packet_environment import packets
#import time
import sys
import matplotlib.pyplot as plt 

num_flies = 100
R_max = 300
pos_theta_min = -np.arctan(1/4)
pos_theta_max = np.arctan(1/4)

dt = 1/60
total_t = 10

num_steps = int(total_t/dt)

seed = int(sys.argv[1])

rand_gen = np.random.RandomState(seed)

"""

def full_antenna(std_box, theta, px, py, a=1.5, b=5.5, dist=0):

		ellipse_val_1 = (np.cos(theta*np.pi/180)*std_box[:,0] + np.sin(theta*np.pi/180)*std_box[:,1])**2/(a**2)

		ellipse_val_2 = (-np.sin(theta*np.pi/180)*std_box[:,0] + np.cos(theta*np.pi/180)*std_box[:,1])**2/(b**2)

		ellipse_points = std_box[ellipse_val_1+ellipse_val_2 <= 1]

		s_l_cond_1 = (theta+180) >= (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360

		s_l_cond_2 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 >= theta

		s_r_cond_1 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 <= theta

		s_r_cond_2 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 >= (theta+180)


		b_l_cond_1 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 >= theta

		b_l_cond_2 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 <= theta-180

		b_r_cond_1 = theta -180 <= (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360

		b_r_cond_2 = (np.arctan2(ellipse_points[:,1],ellipse_points[:,0])*180/np.pi)%360 <= theta


		if 0 <= theta <= 180:

			left_box = ellipse_points[s_l_cond_1*s_l_cond_2]

			right_box = ellipse_points[s_r_cond_1+s_r_cond_2]


		if theta >= 180:

			left_box = ellipse_points[b_l_cond_1 + b_l_cond_2]

			right_box = ellipse_points[b_r_cond_1*b_r_cond_2]


		left_box = np.append(left_box,[[0,0]], axis = 0)
		right_box = np.append(right_box,[[0,0]], axis = 0)

		left_box = np.unique(left_box, axis=0)
		right_box = np.unique(right_box, axis=0)

		translation_x = np.rint(dist*np.cos(theta*np.pi/180))

		translation_y = np.rint(dist*np.sin(theta*np.pi/180))

		t_v_x = px + translation_x

		t_v_y = py + translation_y

		left_box = np.array([t_v_x,t_v_y]) + left_box

		right_box = np.array([t_v_x,t_v_y]) + right_box

		return left_box, right_box

"""

#cols: intensity, gradient, motion, adaptive_int, adaptive_freq, adaptive_tL, adaptive_inst_tt, cos(theta), sin(theta), x, y
#total data is num_flies * num_obs * num_steps (3d array indexed that way-then can flatten along last axis)


def get_rect_antenna(left_std_box, right_std_box, pos, theta):

	num_pts = np.shape(left_std_box)[0]
	
	left_pts = np.zeros(np.shape(left_std_box))
	right_pts = np.zeros(np.shape(right_std_box))

	left_pts[:,0] = np.cos(theta)*left_std_box[:,0] - np.sin(theta)*left_std_box[:,1]
	left_pts[:,1] = np.sin(theta)*left_std_box[:,0] + np.cos(theta)*left_std_box[:,1]

	right_pts[:,0] = np.cos(theta)*right_std_box[:,0] - np.sin(theta)*right_std_box[:,1]
	right_pts[:,1] = np.sin(theta)*right_std_box[:,0] + np.cos(theta)*right_std_box[:,1]

	pos_arr = np.tile(pos, (num_pts,1))

	#print(pos_arr)

	left_pts = left_pts + pos_arr
	right_pts = right_pts + pos_arr

	return left_pts, right_pts


mm_per_px = 0.154

antenna_width = 0.25 
antenna_height = 1
antenna_dist = 0

"""

std_box = np.array([[0,0]])

for i in range(0,int(antenna_height)):

    for j in range(0,int(antenna_height)):

        m = i - (int(antenna_height/2)+1) 
        n = j - (int(antenna_height/2)+1)

        std_box = np.append(std_box, [[m,n]], axis = 0)

std_box = std_box[1:]

"""

def make_L_R_std_box(mm_per_px, antenna_height_mm, antenna_width_mm):

	#makes left and right std boxes for antenna, to be translated and rotated. Assumes orientation is 0 degrees
	#height is long axis of box, tyipcally. 
	#If even then split left and right evenly. If odd then share middle. 

	total_height = np.rint(antenna_height_mm/mm_per_px).astype(int)
	total_width = np.rint(antenna_width_mm/mm_per_px).astype(int)

	x_coords = np.linspace(0,total_width, num = total_width)*mm_per_px
	y_coords = np.linspace(-total_height/2, total_height/2, num = total_height)*mm_per_px
	
	big_box = np.zeros((total_height*total_width,2))

	for i in range(0,total_height):

		for j in range(0,total_width):

			row_idx = i*total_width + j

			big_box[row_idx,0] = x_coords[j]
			big_box[row_idx,1] = np.flip(y_coords)[i]


	#all_x = big_box[:,0]
	all_y = big_box[:,1]

	left_bool = all_y >= 0
	right_bool = all_y <= 0 

	left_box = big_box[left_bool,:]
	right_box = big_box[right_bool,:]


	return left_box, right_box



left_box, right_box = make_L_R_std_box(mm_per_px, antenna_height, antenna_width)

"""

left_x = left_box[:,0]
right_x = right_box[:,0]

left_y = left_box[:,1]
right_y = right_box[:,1]

fig = plt.figure(figsize=(4,4))
plt.scatter(left_x, left_y)
plt.scatter(right_x, right_y)
plt.xlim([0,2])
plt.ylim([-1,1])
plt.show()


left_pts, right_pts = get_rect_antenna(left_box, right_box, pos=np.array([150,90]), theta = np.pi/180*135)

left_x = left_pts[:,0]
right_x = right_pts[:,0]

left_y = left_pts[:,1]
right_y = right_pts[:,1]

plt.scatter(left_x, left_y)
plt.scatter(right_x, right_y)
plt.show()

"""

dw_speed = 60
cw_speed = dw_speed/3

env = packets(dw_speed = dw_speed, source_x=0, source_y=0, rate = 10, cw_type = 'O-U', 
	  corr_lambda = 0.5, cw_speed = cw_speed, delta_t=dt, max_x = 2*R_max, delay_steps = int(2*R_max/(dw_speed*dt)))


###set all fly x,y,theta

all_r = R_max * np.sqrt(rand_gen.uniform(size=num_flies))
all_pos_theta = rand_gen.uniform(low = pos_theta_min, high=pos_theta_max, size = num_flies)

all_x = all_r*np.cos(all_pos_theta)
all_y = all_r*np.sin(all_pos_theta)

all_fly_theta = rand_gen.uniform(low = 0, high = 2*np.pi, size = num_flies) #note that this is in degrees

adaptive_sig = np.zeros(num_flies)

adaptive_timescale = 5

#adaptive_steps = 1

adaptive_steps = int(adaptive_timescale/dt)

###setting adaptive thresh values

for i in range(0,adaptive_steps):
	
	odor_L = np.zeros(num_flies)
	odor_R = np.zeros(num_flies)

	packet_pos, packet_sizes = env.generate_packets(delta_t = dt, rand_gen = rand_gen)

	for j in range(0, num_flies):

		pos = np.array([all_x[j], all_y[j]])

		left_pts, right_pts = get_rect_antenna(left_std_box = left_box, right_std_box = right_box, pos = pos, theta = all_fly_theta[j])                

		odor_L[j], odor_R[j] = env.compute_sig(left_points = left_pts, right_points = right_pts, packet_pos = packet_pos, 
		  packet_sizes = packet_sizes, rand_gen = rand_gen)

	odor = (odor_L + odor_R)/2

	adaptive_sig = adaptive_sig* dt/adaptive_timescale*(odor-adaptive_sig)



all_data = np.zeros((num_flies, 11, num_steps), dtype = np.float32)

whiff_on = np.zeros(num_flies).astype(bool)

odor_bin_prev = np.zeros(num_flies).astype(bool)

odor_L_series = np.zeros((num_steps, num_flies))
odor_R_series = np.zeros((num_steps, num_flies))

all_int = np.zeros(num_flies)
all_freq = np.zeros(num_flies)
all_t_whiff = np.zeros(num_flies)+np.nan
all_inst_tt = np.zeros(num_flies)

#print("running large check")

odor_filt = 0
grad_filt = 0
mot_filt = 0

tau = 1

filter_all = True

print("starting collection")

for i in range(0,num_steps):
	
	x_idx = np.rint(all_x/mm_per_px)
	y_idx = np.rint(all_y/mm_per_px)

	odor_L = np.zeros(num_flies)
	odor_R = np.zeros(num_flies)

	packet_pos, packet_sizes = env.generate_packets(delta_t = dt, rand_gen = rand_gen)

	for j in range(0, num_flies):

		pos = np.array([all_x[j], all_y[j]])

		left_pts, right_pts = get_rect_antenna(left_std_box = left_box, right_std_box = right_box, pos = pos, theta = all_fly_theta[j])                

		odor_L[j], odor_R[j] = env.compute_sig(left_points = left_pts, right_points = right_pts, packet_pos = packet_pos, 
		  packet_sizes = packet_sizes, rand_gen = rand_gen)             

	odor_L_series[i] = odor_L
	odor_R_series[i] = odor_R

	odor = (odor_L + odor_R)/2
	grad = odor_L - odor_R
	grad_filt = grad_filt + dt * (grad-grad_filt)/tau 
	odor_filt = odor_filt + dt*(odor-odor_filt)/tau


	adaptive_sig = adaptive_sig * dt/adaptive_timescale*(odor-adaptive_sig)

	if i == 0:

		mot = 0

	else:

		mot = odor_L_series[i-1]*odor_R_series[i] - odor_R_series[i-1]*odor_L_series[i]


	mot_filt = mot_filt + dt/tau*(mot - mot_filt)

	if filter_all:

		all_data[:,0,i] = odor_filt
		all_data[:,1,i] = grad_filt
		all_data[:,2,i] = mot_filt

	else:

		all_data[:,0,i] = odor
		all_data[:,1,i] = grad
		all_data[:,2,i] = mot

	adaptive_thresh = adaptive_sig/2
	one_sig = np.ones(num_flies)

	true_thresh = np.maximum(one_sig, adaptive_thresh)

	odor_bin = odor > true_thresh

	all_int = all_int + dt/tau*(odor_bin.astype(float)-all_int)

	all_data[:,3,i] = all_int

	new_whiffs = odor_bin * (~odor_bin_prev)

	#print(np.sum(new_whiffs))

	all_freq[~new_whiffs] = all_freq[~new_whiffs]*np.exp(-dt/tau)
	all_freq[new_whiffs] = all_freq[new_whiffs]*np.exp(-dt/tau) + 1

	all_data[:,4,i] = all_freq

	t_now = i*dt

	all_t_whiff[new_whiffs] = t_now

	all_tl = t_now - all_t_whiff

	all_data[:,5,i] = all_tl

	all_inst_tt[odor_bin] = 1

	all_inst_tt[~odor_bin] = all_inst_tt[~odor_bin]*np.exp(-dt/tau)

	all_data[:,6,i] = all_inst_tt

	all_data[:,7,i] = np.cos(all_fly_theta*np.pi/180)
	all_data[:,8,i] = np.sin(all_fly_theta*np.pi/180)
	all_data[:,9,i] = all_x
	all_data[:,10,i] = all_y

	odor_bin_prev = odor_bin


np.save(str(seed)+"_all_data.npy", all_data)



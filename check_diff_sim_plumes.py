import numpy as np 
from packet_environment import packets
#import time
import sys

num_flies = 1000
R_max = 300
pos_theta_min = -np.arctan(1/4)
pos_theta_max = np.arctan(1/4)

dt = 1/60
total_t = 10

num_steps = int(total_t/dt)

seed = int(sys.argv[1])

rand_gen = np.random.RandomState(seed)


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


#cols: intensity, gradient, motion, adaptive_int, adaptive_freq, adaptive_tL, adaptive_inst_tt, cos(theta), sin(theta), x, y
#total data is num_flies * num_obs * num_steps (3d array indexed that way-then can flatten along last axis)


mm_per_px = 0.154

antenna_width = 0.5 * 1/mm_per_px
antenna_height = 1.5 * 1/mm_per_px
antenna_dist = 0

std_box = np.array([[0,0]])

for i in range(0,int(antenna_height)+2):

    for j in range(0,int(antenna_height)+2):

        m = i - (int(antenna_height/2)+1) 
        n = j - (int(antenna_height/2)+1)

        std_box = np.append(std_box, [[m,n]], axis = 0)

std_box = std_box[1:]

dw_speed = 60
cw_speed = dw_speed/3

env = packets(dw_speed = dw_speed, source_x=0, source_y=0, rate = 10, cw_type = 'O-U', 
	  corr_lambda = 0.5, cw_speed = cw_speed, delta_t=dt, max_x = 2*R_max, delay_steps = int(2*R_max/(dw_speed*dt)))


###set all fly x,y,theta

all_r = R_max * np.sqrt(rand_gen.uniform(size=num_flies))
all_pos_theta = rand_gen.uniform(low = pos_theta_min, high=pos_theta_max, size = num_flies)

all_x = all_r*np.cos(all_pos_theta)
all_y = all_r*np.sin(all_pos_theta)

all_fly_theta = rand_gen.uniform(low = 0, high = 360, size = num_flies) #note that this is in degrees

adaptive_sig = np.zeros(num_flies)

adaptive_timescale = 5

#adaptive_steps = 1

adaptive_steps = int(adaptive_timescale/dt)

###setting adaptive thresh values

for i in range(0,adaptive_steps):
	
	x_idx = np.rint(all_x/mm_per_px)
	y_idx = np.rint(all_y/mm_per_px)

	odor_L = np.zeros(num_flies)
	odor_R = np.zeros(num_flies)

	packet_pos, packet_sizes = env.generate_packets(delta_t = dt, rand_gen = rand_gen)

	for j in range(0, num_flies):

		left_box, right_box = full_antenna(std_box, all_fly_theta[j], x_idx[j], y_idx[j], a=antenna_width/2, b=antenna_height/2, dist=antenna_dist)
	                    
		left_box = mm_per_px * left_box

		right_box = mm_per_px * right_box                 

		odor_L[j], odor_R[j] = env.compute_sig(left_points = left_box, right_points = right_box, packet_pos = packet_pos, 
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

for i in range(0,num_steps):
	
	x_idx = np.rint(all_x/mm_per_px)
	y_idx = np.rint(all_y/mm_per_px)

	odor_L = np.zeros(num_flies)
	odor_R = np.zeros(num_flies)

	packet_pos, packet_sizes = env.generate_packets(delta_t = dt, rand_gen = rand_gen)

	for j in range(0, num_flies):

		left_box, right_box = full_antenna(std_box, all_fly_theta[j], x_idx[j], y_idx[j], a=antenna_width/2, b=antenna_height/2, dist=antenna_dist)
	                    
		left_box = mm_per_px * left_box

		right_box = mm_per_px * right_box                 

		odor_L[j], odor_R[j] = env.compute_sig(left_points = left_box, right_points = right_box, packet_pos = packet_pos, 
		  packet_sizes = packet_sizes, rand_gen = rand_gen)

	odor_L_series[i] = odor_L
	odor_R_series[i] = odor_R

	odor = (odor_L + odor_R)/2

	adaptive_sig = adaptive_sig * dt/adaptive_timescale*(odor-adaptive_sig)

	all_data[:,0,i] = odor
	all_data[:,1,i] = odor_L-odor_R

	if i == 0:

		all_data[:,2,i] = 0

	else:

		all_data[:,2,i] = odor_L_series[i-1]*odor_R_series[i] - odor_R_series[i-1]*odor_L_series[i]


	adaptive_thresh = adaptive_sig/2
	one_sig = np.ones(num_flies)

	true_thresh = np.maximum(one_sig, adaptive_thresh)

	odor_bin = odor > true_thresh

	all_int = all_int + dt/2*(odor_bin.astype(float)-all_int)

	all_data[:,3,i] = all_int

	new_whiffs = odor_bin * (~odor_bin_prev)

	#print(np.sum(new_whiffs))

	all_freq[~new_whiffs] = all_freq[~new_whiffs]*np.exp(-dt/2)
	all_freq[new_whiffs] = all_freq[new_whiffs]*np.exp(-dt/2) + 1

	all_data[:,4,i] = all_freq

	t_now = i*dt

	all_t_whiff[new_whiffs] = t_now

	all_tl = t_now - all_t_whiff

	all_data[:,5,i] = all_tl

	all_inst_tt[odor_bin] = 1

	all_inst_tt[~odor_bin] = all_inst_tt[~odor_bin]*np.exp(-dt/2)

	all_data[:,6,i] = all_inst_tt

	all_data[:,7,i] = np.cos(all_fly_theta*np.pi/180)
	all_data[:,8,i] = np.sin(all_fly_theta*np.pi/180)
	all_data[:,9,i] = all_x
	all_data[:,10,i] = all_y

	odor_bin_prev = odor_bin


np.save(str(seed)+"_all_data.npy", all_data)




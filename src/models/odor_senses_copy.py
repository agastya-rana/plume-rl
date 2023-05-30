import numpy as np 
import copy

#probably can make it inherit from packets class? Or make one class for videos and another for simulated plumes

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


class OdorFeatures():

	def __init__(self, config):
		self.dt = config['DELTA_T_S']
		self.clear()
		self.std_left_box, self.std_right_box = make_L_R_std_box(mm_per_px = config['MM_PER_PX'], antenna_height_mm = config['ANTENNA_LENGTH_MM'], antenna_width_mm = config['ANTENNA_WIDTH_MM'])
		self.mm_per_px = config['MM_PER_PX']
		self.use_movie = config['USE_MOVIE']
		self.num_pts = np.shape(self.std_left_box)[0]
		self.tau = config['TEMPORAL_FILTER_TIMESCALE_S']
		self.adaptation_tau = config['TEMPORAL_THRESHOLD_ADAPTIVE_TIMESCALE_S']
		self.filter_all = config['TEMPORAL_FILTER_ALL']
		self.threshold_style = config['CONCENTRATION_THRESHOLD_STYLE']
		self.base_threshold = config['CONCENTRATION_BASE_THRESHOLD']
		self.max_conc = config['MAX_CONCENTRATION']
		self.max_hrc = self.max_conc**2
		self.max_t_L = self.dt*config['STOP_FRAME']
		self.normalize = config['NORMALIZE_ODOR_FEATURES']

		#other temporal ones here too

	def _rotate_and_translate_sensors(self, theta, pos):

		self.left_pts = np.zeros(np.shape(self.std_left_box))
		self.right_pts = np.zeros(np.shape(self.std_right_box))

		self.left_pts[:,0] = np.cos(theta)*self.std_left_box[:,0] - np.sin(theta)*self.std_left_box[:,1]
		self.left_pts[:,1] = np.sin(theta)*self.std_left_box[:,0] + np.cos(theta)*self.std_left_box[:,1]

		self.right_pts[:,0] = np.cos(theta)*self.std_right_box[:,0] - np.sin(theta)*self.std_right_box[:,1]
		self.right_pts[:,1] = np.sin(theta)*self.std_right_box[:,0] + np.cos(theta)*self.std_right_box[:,1]

		pos_arr = np.tile(pos, (self.num_pts,1))

		self.left_pts = self.left_pts + pos_arr
		self.right_pts = self.right_pts + pos_arr


	def _get_left_right_odors(self, odor_frame = None):

		self.left_odors = np.zeros(self.num_pts)
		self.right_odors = np.zeros(self.num_pts)

		self.left_idxs = np.rint(self.left_pts/self.mm_per_px).astype(int)
		self.right_idxs = np.rint(self.right_pts/self.mm_per_px).astype(int)

		for i in range(0,self.num_pts):

			if self.use_movie:

				try: 
					self.left_odors[i] = odor_frame[self.left_idxs[i,0], self.left_idxs[i,1]]
				except IndexError:
					self.left_odors[i] = 0

				try:
					self.right_odors[i] = odor_frame[self.right_idxs[i,0], self.right_idxs[i,1]]
				except IndexError:
					self.right_odors[i] = 0

		self.mean_left_odor = np.mean(self.left_odors)
		self.mean_right_odor = np.mean(self.right_odors)


	def update(self, theta, pos, odor_frame):

		self._rotate_and_translate_sensors(theta = theta, pos = pos)
		self._get_left_right_odors(odor_frame=odor_frame)
		self.concentration = (1/2)*(self.mean_left_odor+self.mean_right_odor)
		self.hrc = self.left_odor_prev*self.mean_right_odor - self.right_odor_prev*self.mean_left_odor
		self.gradient = self.mean_left_odor - self.mean_right_odor

		#DO TEMPORAL ONES

		if self.threshold_style == 'fixed':

			self.odor_bin = self.concentration > self.base_threshold

		elif self.threshold_style == 'adaptive':

			self.adaptation = self.dt/self.adaptation_tau*(self.concentration-self.adaptation)
			self.threshold = np.maximum(self.base_threshold, self.adaptation)
			self.odor_bin = self.concentration > self.threshold 
		
		self.intermittency += 1/self.tau*(self.odor_bin-self.intermittency)*self.dt

		self.new_whiff = self.odor_bin * (not(self.odor_bin_prev))

		self.t_now += self.dt

		if self.new_whiff:

			self.t_whiff = copy.deepcopy(self.t_now)

		self.t_L_arr[0] = self.t_L
		self.t_L = self.t_now - self.t_whiff
		self.t_L_arr[1] = self.t_L

		if self.normalize:

			self.adaptation = self.adaptation/self.max_conc
			self.concentration = self.concentration/self.max_conc
			self.gradient = self.gradient/self.max_conc
			self.hrc = self.hrc/self.max_hrc
			self.t_L_arr = self.t_L_arr/self.max_t_L


		if self.filter_all:

			self.filt_conc = self.dt/self.tau*(self.concentration-self.filt_conc)
			self.filt_grad = self.dt/self.tau*(self.gradient-self.filt_grad)
			self.filt_hrc = self.dt/self.tau*(self.hrc-self.filt_hrc)

		
		#UPDATE PREV VALUES FOR NEXT STEP

		self.left_odor_prev = self.mean_left_odor
		self.right_odor_prev = self.mean_right_odor
		self.odor_prev = self.concentration
		self.grad_prev = self.gradient
		self.hrc_prev = self.hrc
		self.odor_bin_prev = self.odor_bin

		#RETURN 


		if self.filter_all:

			return np.array([self.filt_conc, self.filt_grad, self.filt_hrc, self.intermittency, self.t_L_arr[0], self.t_L_arr[1]])

		else:

			return np.array([self.concentration, self.gradient, self.hrc, self.intermittency, self.t_L_arr[0], self.t_L_arr[1]])


	def clear(self):

		self.left_odor_prev = 0
		self.right_odor_prev = 0
		self.odor_prev = 0
		self.odor_bin_prev = False
		self.t_L = 100.
		self.intermittency = 0
		self.t_now = 0
		self.concentration = 0
		self.gradient = 0
		self.hrc = 0
		self.grad_prev = 0
		self.hrc_prev = 0
		self.conc_prev = 0
		self.filt_conc = 0
		self.filt_grad = 0
		self.filt_hrc = 0
		self.adaptation = 0
		self.t_whiff = -100.
		self.t_L_arr = np.array([100.,100+self.dt])



class OdorFeatures_no_temporal()


	def __init__(self, config):
		self.dt = config['DELTA_T_S']
		self.clear()
		self.std_left_box, self.std_right_box = make_L_R_std_box(mm_per_px = config['MM_PER_PX'], antenna_height_mm = config['ANTENNA_LENGTH_MM'], antenna_width_mm = config['ANTENNA_WIDTH_MM'])
		self.mm_per_px = config['MM_PER_PX']
		self.use_movie = config['USE_MOVIE']
		self.num_pts = np.shape(self.std_left_box)[0]
		self.base_threshold = config['CONCENTRATION_BASE_THRESHOLD']
		self.max_conc = config['MAX_CONCENTRATION']
		self.max_hrc = self.max_conc**2
		self.normalize = config['NORMALIZE_ODOR_FEATURES']
		self.discretize = config['DISCRETIZE_ODOR_FEATURES']

		if self.normalize:

			self.odor_threshold = self.base_threshold/self.max_conc

		else:

			self.odor_threshold = self.base_threshold



	def _rotate_and_translate_sensors(self, theta, pos):

		self.left_pts = np.zeros(np.shape(self.std_left_box))
		self.right_pts = np.zeros(np.shape(self.std_right_box))

		self.left_pts[:,0] = np.cos(theta)*self.std_left_box[:,0] - np.sin(theta)*self.std_left_box[:,1]
		self.left_pts[:,1] = np.sin(theta)*self.std_left_box[:,0] + np.cos(theta)*self.std_left_box[:,1]

		self.right_pts[:,0] = np.cos(theta)*self.std_right_box[:,0] - np.sin(theta)*self.std_right_box[:,1]
		self.right_pts[:,1] = np.sin(theta)*self.std_right_box[:,0] + np.cos(theta)*self.std_right_box[:,1]

		pos_arr = np.tile(pos, (self.num_pts,1))

		self.left_pts = self.left_pts + pos_arr
		self.right_pts = self.right_pts + pos_arr


	def _get_left_right_odors(self, odor_frame = None):

		self.left_odors = np.zeros(self.num_pts)
		self.right_odors = np.zeros(self.num_pts)

		self.left_idxs = np.rint(self.left_pts/self.mm_per_px).astype(int)
		self.right_idxs = np.rint(self.right_pts/self.mm_per_px).astype(int)

		for i in range(0,self.num_pts):

			if self.use_movie:

				try: 
					self.left_odors[i] = odor_frame[self.left_idxs[i,0], self.left_idxs[i,1]]
				except IndexError:
					self.left_odors[i] = 0

				try:
					self.right_odors[i] = odor_frame[self.right_idxs[i,0], self.right_idxs[i,1]]
				except IndexError:
					self.right_odors[i] = 0

		self.mean_left_odor = np.mean(self.left_odors)
		self.mean_right_odor = np.mean(self.right_odors)


	def update(self, theta, pos, odor_frame):

		self._rotate_and_translate_sensors(theta = theta, pos = pos)
		self._get_left_right_odors(odor_frame=odor_frame)
		self.concentration = (1/2)*(self.mean_left_odor+self.mean_right_odor)
		self.hrc = self.left_odor_prev*self.mean_right_odor - self.right_odor_prev*self.mean_left_odor
		self.gradient = self.mean_left_odor - self.mean_right_odor

		if self.normalize:

			self.concentration = self.concentration/self.max_conc
			self.gradient = self.gradient/self.max_conc
			self.hrc = self.hrc/self.max_hrc

			self.comparison_mean_left_odor = self.mean_left_odor/self.max_conc
			self.comparison_mean_right_odor = self.mean_right_odor/self.max_conc

		else:

			self.comparison_mean_left_odor = self.mean_left_odor
			self.comparison_mean_right_odor = self.mean_right_odor


		if self.discretize:

			self._discretize_fn()


		self.left_odor_prev = self.mean_left_odor
		self.right_odor_prev = self.mean_right_odor
		self.odor_prev = self.concentration
		self.grad_prev = self.gradient
		self.hrc_prev = self.hrc

		return np.array([self.concentration, self.gradient, self.hrc])


	def _discretize_fn(self):

		self.concentration = (self.concentration > self.odor_threshold).astype(int)

		#for gradient and motion:

		if (self.comparison_mean_left_odor < self.odor_threshold) and (self.comparison_mean_right_odor < self.odor_threshold):

			self.hrc = 0
			self.gradient = 0

		else:

			if self.gradient > 0:

				self.gradient = 1

			else:

				self.gradient = 2

			if self.hrc > 0:

				self.hrc = 1

			else:

				self.hrc = 2







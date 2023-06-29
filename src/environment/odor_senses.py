import numpy as np 
import copy

class OdorFeatures():

	def __init__(self, config):
		"""
		The following features have been implemented with their corresponding keys:
		- concentration (conc)
		- gradient (grad)
		- hrc (hrc)
		- concentration_discrete (conc_disc)
		- gradient_discrete (grad_disc)
		- hrc_discrete (hrc_disc)
		- concentration_left (conc_left)
		- concentration_right (conc_right)
		- intermittency (intermittency)
		- t_L (t_L)		

		Note that we assume that the plume is normalized to a max intensity of 1. 
		We also assume that all our features will be normalized to a max of 1 (min of 0 or -1 depending on the feature).
		Therefore, the detection_threshold must be between 0 and 1 (default is 0).
		"""

		agent_dict = config['agent']
		plume_dict = config['plume']
		state_dict = config['state']
		self.dt = None ## Set by reset depending on plume used
		self.features = state_dict['FEATURES']
		if state_dict["DISCRETE_OBSERVABLES"]:
			assert set(self.features).issubset(set(['conc_disc', 'grad_disc', 'hrc_disc'])), "Can only discretize concentration, gradient, and hrc"
		
		## Initialize base threshold
		self.detection_threshold = state_dict['DETECTION_THRESHOLD'] if "DETECTION_THRESHOLD" in state_dict else 0
		self.threshold_type = state_dict["DETECTION_THRESHOLD_TYPE"] if "DETECTION_THRESHOLD_TYPE" in state_dict else "fixed"
		if self.threshold_type == "adaptive":
			self.threshold_adaptation_timescale = state_dict["DETECTION_THRESHOLD_TIMESCALE_S"]
		
		self.tau = state_dict['TAU_S'] if "TAU_S" in state_dict else 1 ## Time constant for exponential moving average used in intermittency calculation
		if "MAX_T_L_S" in state_dict:
			self.max_t_L = state_dict['MAX_T_L_S']
		elif "t_L" in self.features:
			raise Exception("Must specify MAX_T_L if using t_L feature")
		self.fix_antenna = state_dict['FIX_ANTENNA'] if "FIX_ANTENNA" in state_dict else False

		feature_func_map = {'conc': self.get_conc, 'grad': self.get_grad, 'hrc': self.get_hrc, 'conc_left': self.get_conc_left, 'conc_right': self.get_conc_right, 'intermittency': self.get_intermittency, 't_L': self.get_t_L, 
		'conc_disc': self.get_conc_disc, 'grad_disc': self.get_grad_disc, 'hrc_disc': self.get_hrc_disc}
		bounds_func_map = {'conc': [0, 1], 'grad': [-1, 1], 'hrc': [-1, 1], 'conc_left': [0, 1], 'conc_right': [0, 1], 'intermittency': [0, 1], 't_L': [0, 1]}

		## Set box for odor feature detection
		self.mm_per_px = None ## Set by reset depending on plume used
		self.std_left_box, self.std_right_box = None, None ## Set by reset depending on plume used
		self.num_pts = None ## Set by reset depending on plume used

		## Set up feature functions and bounds
		self.func_evals = [feature_func_map[feat] for feat in self.features]
		self.feat_bounds = np.array([bounds_func_map[feat] for feat in self.features])
		num_discrete = lambda x : 3 if x in ['grad_disc', 'hrc_disc'] else 2
		self.discretization_index = [num_discrete(f) for f in self.features if f in ['conc_disc', 'grad_disc', 'hrc_disc']]
		
		## Reset feature values
		self.clear()


	@staticmethod
	def make_L_R_std_box(mm_per_px, antenna_height_mm, antenna_width_mm):
		"""
		Make a standard left and right box for the antenna that can be rotated and translated to match the orientation and position of the antenna.
		The boxes have dimension (px_height*px_width, 2) where px_height and px_width are the height and width of the antenna in pixels.
		x,y coordinates are in units of mm.
		"""
		## Height is long axis of box, typically.
		## If even then split left and right evenly. If odd then share middle. 

		px_height = round(antenna_height_mm/mm_per_px)
		px_width = round(antenna_width_mm/mm_per_px)
		x_coords = np.linspace(0, px_width,px_width)*mm_per_px
		y_coords = np.flip(np.linspace(-px_height/2, px_height/2,px_height)*mm_per_px)

		## Make a big box with meshgrid.
		big_box = np.zeros((px_height*px_width,2))
		x_grid, y_grid = np.meshgrid(x_coords, y_coords)
		big_box = np.column_stack((x_grid.flatten(), y_grid.flatten()))

		# Separate the box into left and right halves
		left_box = big_box[big_box[:, 1] >= 0]
		right_box = big_box[big_box[:, 1] <= 0]
		return left_box, right_box
	
	@staticmethod
	def _rotate_points(points, theta):
		## Assumes that points is a 2D array with x and y coordinates in the first and second columns, respectively.
		x = np.cos(theta) * points[:, 0] - np.sin(theta) * points[:, 1]
		y = np.sin(theta) * points[:, 0] + np.cos(theta) * points[:, 1]
		return np.column_stack((x, y))

	def _rotate_and_translate_sensors(self, theta, pos):
		pos_arr = np.tile(pos, (self.num_pts,1))
		self.left_pts = self._rotate_points(self.std_left_box, theta) + pos_arr
		self.right_pts = self._rotate_points(self.std_right_box, theta) + pos_arr

	def _get_left_right_odors(self, odor_frame = None):
		self.left_odors = np.zeros(self.num_pts)
		self.right_odors = np.zeros(self.num_pts)
		self.left_idxs = np.rint(self.left_pts/self.mm_per_px).astype(int)
		self.right_idxs = np.rint(self.right_pts/self.mm_per_px).astype(int)

		try:
			self.left_odors = odor_frame[self.left_idxs[:,0], self.left_idxs[:,1]]
		except IndexError:
			self.left_odors = np.zeros(self.num_pts) ## If the agent is out of bounds, then the odor is zero.
		try:
			self.right_odors = odor_frame[self.right_idxs[:,0], self.right_idxs[:,1]]
		except IndexError:
			self.right_odors = np.zeros(self.num_pts)

		self.mean_left_odor = np.mean(self.left_odors)
		self.mean_right_odor = np.mean(self.right_odors)
		if self.detection_threshold:
			self.mean_left_odor = self.mean_left_odor*(self.mean_left_odor>self.detection_threshold)
			self.mean_right_odor = self.mean_right_odor*(self.mean_right_odor>self.detection_threshold)
		self.concentration = (self.mean_left_odor + self.mean_right_odor)/2

	def get_conc_left(self, normalize=False):
		return self.mean_left_odor
	
	def get_conc_right(self, normalize=False):
		return self.mean_right_odor
	
	def get_conc(self, normalize=False):
		return (self.mean_left_odor + self.mean_right_odor)/2
	
	def get_grad(self, normalize=False):
		return self.mean_left_odor - self.mean_right_odor

	def get_hrc(self, normalize=False):
		return self.left_odor_prev*self.mean_right_odor - self.right_odor_prev*self.mean_left_odor
	
	def get_intermittency(self, normalize=False):
		assert self.tau is not None, "tau must be set to use intermittency feature."
		self.intermittency += 1/self.tau*(self.odor_bin-self.intermittency)*self.dt
		return self.intermittency

	def get_t_L(self, normalize=False):
		return (self.t_now - self.t_whiff)/self.max_t_L
	
	def get_conc_disc(self, normalize=False):
		return int(self.concentration > self.detection_threshold)
	
	def get_grad_disc(self, normalize=False):
		left_disc = int(self.mean_left_odor > self.detection_threshold)
		right_disc = int(self.mean_right_odor > self.detection_threshold)
		return left_disc - right_disc + 1

	def get_hrc_disc(self, normalize=False):
		left_disc = int(self.mean_left_odor > self.detection_threshold)
		right_disc = int(self.mean_right_odor > self.detection_threshold)
		left_disc_prev = int(self.left_odor_prev > self.detection_threshold)
		right_disc_prev = int(self.right_odor_prev > self.detection_threshold)
		return (left_disc_prev*right_disc - right_disc_prev*left_disc) + 1
		
	def update(self, theta, pos, odor_frame):
		if self.fix_antenna:
			theta = np.pi ## Rotate the antenna from downwind to upwind.
		## Returns the odor features requested in the order of state_dict['features']
		self._rotate_and_translate_sensors(theta = theta, pos = pos)
		self._get_left_right_odors(odor_frame=odor_frame)
		return self.get_features()

	def get_features(self):
		self.update_bins()
		self.update_whiff()
		feats = np.array([self.func_evals[i](normalize=self.normalize) for i in range(len(self.func_evals))])
		self.update_hist()
		return feats	
	
	def update_bins(self):
		if self.threshold_style == 'fixed':
			self.odor_bin = self.concentration > self.detection_threshold
		elif self.threshold_style == 'adaptive':
			self.adaptation += self.dt/self.threshold_adaptation_timescale*(self.concentration-self.adaptation)
			self.threshold = np.maximum(self.detection_threshold, self.adaptation)
			self.odor_bin = self.concentration > self.threshold

	def update_whiff(self):
		self.new_whiff = self.odor_bin * (not(self.odor_bin_prev))
		self.t_now += self.dt
		if self.new_whiff:
			self.t_whiff = copy.deepcopy(self.t_now)

	def update_hist(self):
		self.left_odor_prev = self.mean_left_odor
		self.right_odor_prev = self.mean_right_odor
		self.odor_prev = self.concentration
		#self.grad_prev = self.gradient
		#self.hrc_prev = self.hrc
		self.odor_bin_prev = self.odor_bin

	def clear(self):
		## Reset all values
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
		#self.grad_prev = 0
		#self.hrc_prev = 0
		self.adaptation = 0
		self.t_whiff = -100.
		self.t_L = 1000.
		self.odor_bin = False

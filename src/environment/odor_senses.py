import numpy as np 
import copy
import scipy
from src.packets.packet_environment import packets

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
		self.discrete_observables = state_dict['DISCRETE_OBSERVABLES'] if "DISCRETE_OBSERVABLES" in state_dict else False
		if self.discrete_observables:
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
		self.init_intensity = None #only used when computing signal in packet sims. Will get set in reset method in gym environment.
		self.plume_type = 'movie' #this may get changed in reset method due to plume_switching

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

	def _compute_sig_from_packet_sim(self, left_points, right_points, packet_pos, packet_sizes):
		all_points = np.vstack((left_points, right_points))
		all_distances = scipy.spatial.distance_matrix(all_points, packet_pos)
		scaled_all_distances = all_distances/(packet_sizes[None,:])
		gaussian_part = np.exp(-(scaled_all_distances)**2)
		packet_prefactor = self.init_intensity/(np.pi*packet_sizes**2)
		all_signals = gaussian_part * packet_prefactor[None, :]
		all_total_signals = np.sum(all_signals, axis = 1)
		total_left_sig = all_total_signals[0:len(left_points[:,0])]
		total_right_sig = all_total_signals[len(left_points[:,0]):]
		left_sig = np.mean(total_left_sig)
		right_sig = np.mean(total_right_sig)

		return left_sig, right_sig 


	def _get_left_right_odors(self, odor_frame = None):
		self.left_odors = np.zeros(self.num_pts)
		self.right_odors = np.zeros(self.num_pts)
		## NOTE THAT THIS ONLY WORKS FOR POSITIVE POSITION X AND Y VALUES, ELSE THESE INDICES GO NEGATIVE
		if self.plume_type == 'movie' or self.plume_type == 'ribbon':
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

		elif self.plume_type == 'packet_simulation':
			packet_pos = odor_frame[:,0:2]
			packet_sizes = odor_frame[:,2]
			self.left_odors, self.right_odors = self._compute_sig_from_packet_sim(left_points = self.left_pts, right_points = self.right_pts, 
				packet_pos = packet_pos, packet_sizes = packet_sizes)

		self.mean_left_odor = np.mean(self.left_odors)
		self.mean_right_odor = np.mean(self.right_odors)

		if self.detection_threshold:
			self.mean_left_odor = self.mean_left_odor*(self.mean_left_odor>self.detection_threshold)
			self.mean_right_odor = self.mean_right_odor*(self.mean_right_odor>self.detection_threshold)
		self.concentration = (self.mean_left_odor + self.mean_right_odor)/2

	def get_conc_left(self):
		return self.mean_left_odor
	
	def get_conc_right(self):
		return self.mean_right_odor
	
	def get_conc(self):
		return (self.mean_left_odor + self.mean_right_odor)/2
	
	def get_grad(self):
		return self.mean_left_odor - self.mean_right_odor

	def get_hrc(self):
		return self.left_odor_prev*self.mean_right_odor - self.right_odor_prev*self.mean_left_odor
	
	def get_intermittency(self):
		assert self.tau is not None, "tau must be set to use intermittency feature."
		self.intermittency += 1/self.tau*(self.odor_bin-self.intermittency)*self.dt
		return self.intermittency

	def get_t_L(self):
		return (self.t_now - self.t_whiff)/self.max_t_L
	
	def get_conc_disc(self):
		return int(self.concentration > self.detection_threshold)
	
	def get_grad_disc(self):
		left_disc = int(self.mean_left_odor > self.detection_threshold)
		right_disc = int(self.mean_right_odor > self.detection_threshold)
		return left_disc - right_disc + 1

	def get_hrc_disc(self):
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
		feats = np.array([self.func_evals[i]() for i in range(len(self.func_evals))])
		self.update_hist()
		return feats	
	
	def update_bins(self):
		if self.threshold_type == 'fixed':
			self.odor_bin = self.concentration > self.detection_threshold
		elif self.threshold_type == 'adaptive':
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

	
class OdorFeaturesAllocentric(OdorFeatures):
	def __init__(self, config):
		## Set box for odor feature detection
		self.features = ["conc", "gradx", "grady", "hrcx", "hrcy"]
		self.mm_per_px = None ## Set by reset depending on plume used
		self.std_left_box, self.std_right_box = None, None ## Set by reset depending on plume used
		self.num_pts = None ## Set by reset depending on plume used
		self.dt = None ## Set by reset depending on plume used
		## Set up feature functions and bounds
		self.feat_bounds = np.array([[0,1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])
		self.init_intensity = None ## again set by reset method for packet sims
		self.min_wall_x, self.max_wall_x, self.min_wall_y, self.max_wall_y = None, None, None, None ## Set by reset depending on plume used
		## Reset feature values
		self.plume_type = None
		self.clear()
	
	## Allocentric feature functions for gradx, grady, hrcx, hrcy
	## First get the gradient and hrc from the odor frame then use in functions
	def _get_allocentric(self, odor_frame):

		if self.plume_type == "packet_simulation":
			odor_coord = copy.deepcopy(odor_frame)
			## Find the actual odor frame from coordinates of the packets
			## Define the x and y ranges of the odor frame from the positions of left and right boxes - including padding to allow neighbors to work
			x_range = np.arange(np.min(self.all_pts[:,0]) - 1, np.max(self.all_pts[:,0])+2)
			y_range = np.arange(np.min(self.all_pts[:,1]) - 1, np.max(self.all_pts[:,1])+2)
			odor_frame = np.zeros((round(self.max_wall_x/self.mm_per_px) + 1, round(self.max_wall_y/self.mm_per_px) + 1))
			odor_frame[np.min(self.all_pts[:,0]) - 1: np.max(self.all_pts[:,0])+2, np.min(self.all_pts[:,1]) - 1: np.max(self.all_pts[:,1])+2] = packets.compute_odor_at_timestep(x_range, y_range, self.init_intensity, odor_coord)
		if self.odor_prev is None:
			self.odor_prev = np.zeros_like(odor_frame)
		gradx, grady, hrcx, hrcy = 0, 0, 0, 0
		num = len(self.all_pts)
		total_conc = 0
		for x,y in self.all_pts:
			neighbors = {
				'top': (x, y-1),
				'bottom': (x, y+1),
				'left': (x-1, y),
				'right': (x+1, y),
				'top_left': (x-1, y-1),
				'top_right': (x+1, y-1),
				'bottom_left': (x-1, y+1),
				'bottom_right': (x+1, y+1)
			}
			# Initialize velocity and gradient components
			vx, vy = 0, 0
			gx, gy = 0, 0
			# Compute for each neighbor
			central_prev = self.odor_prev[x, y]
			central_signal = odor_frame[x, y]
			total_conc += central_signal
			for direction, (nx, ny) in neighbors.items():
				# Extract odor signals for the current neighbor and the central point
				neighbor_signal = odor_frame[nx, ny]
				neighbor_prev = self.odor_prev[nx, ny]
				motion = central_prev * neighbor_signal - central_signal * neighbor_prev
				gradient = neighbor_signal - central_signal				
				# Project into allocentric coordinates
				if direction == 'top':
					vy += motion
					gy += gradient
				elif direction == 'bottom':
					vy -= motion
					gy -= gradient
				elif direction == 'left':
					vx -= motion
					gx -= gradient
				elif direction == 'right':
					vx += motion
					gx += gradient
				else:  # Diagonal neighbors
					if 'top' in direction:
						vy_factor = 1
					else:
						vy_factor = -1
					if 'left' in direction:
						vx_factor = -1
					else:
						vx_factor = 1
					# Distributing the motion or gradient equally in both x and y directions
					vx += vx_factor * motion / np.sqrt(2)
					vy += vy_factor * motion / np.sqrt(2)
					gx += vx_factor * gradient / np.sqrt(2)
					gy += vy_factor * gradient / np.sqrt(2)
			gradx += gx/num
			grady += gy/num
			hrcx += vx/num
			hrcy += vy/num
		conc = total_conc/num
		self.odor_prev = odor_frame
		return conc, gradx, grady, hrcx, hrcy

	def update(self, theta, pos, odor_frame):
		## Returns the odor features requested in the order of state_dict['features']
		self.left_coor = self.std_left_box + np.tile(pos, (self.num_pts,1))
		self.right_coor = self.std_right_box + np.tile(pos, (self.num_pts,1))
		self.left_idxs = np.rint(self.left_coor/self.mm_per_px).astype(int)
		self.right_idxs = np.rint(self.right_coor/self.mm_per_px).astype(int)
		self.all_pts = np.vstack((self.left_idxs, self.right_idxs)) ## dim = (2*num_pts, 2)
		feats = self._get_allocentric(odor_frame=odor_frame)
		return feats
	
	def compute_sig(self, all_points):

		"""
		Computes odor signal at a given set of locations (all_points). all_points is expected to be an array of size (n,2),
		where first column indicates x-coordinate and second indicates y-coordinate.
		"""

		all_distances = scipy.spatial.distance_matrix(all_points, self.packet_pos_mat) #creates a matrix of size (num_points, num_packets) and stores distance from point i to packet j
		scaled_all_distances = all_distances/(self.packet_sizes[None,:])
		gaussian_part = np.exp(-(scaled_all_distances)**2)
		packet_prefactor = self.init_intensity/(np.pi*self.packet_sizes**2)
		all_signals_per_packet = gaussian_part * packet_prefactor[None, :]
		all_total_signals = np.sum(all_signals_per_packet, axis = 1)

		return all_total_signals
	
	def clear(self):
		## Reset all values
		self.odor_prev = None
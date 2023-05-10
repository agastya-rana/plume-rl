import numpy as np 

#probably can make it inherit from packets class? Or make one class for videos and another for simulated plumes

class OdorFeatures():

	def __init__(self, config):

		self.std_left_box = #fill in
		self.std_right_box = #fill in
		self.config = config
		self.num_pts = np.shape(std_left_box)[0]
		self.left_odor_prev = 0
		self.right_odor_prev = 0
		self.odor_prev = 0
		self.t_L = 0
		self.tau = config['TEMPORAL_FILTER_TIMESCALE_S']
		#other temporal ones here too

	def _rotate_and_translate_sensors(self, theta, pos):

		self.left_pts = np.zeros(np.shape(self.std_left_box))
		self.right_pts = np.zeros(np.shape(self.std_right_box))

		self.left_pts[:,0] = np.cos(theta)*self.std_left_box[:,0] - np.sin(theta)*self.std_left_box[:,1]
		self.left_pts[:,1] = np.sin(theta)*self.std_left_box[:,0] + np.cos(theta)*self.std_left_box[:,1]

		self.right_pts[:,0] = np.cos(theta)*self.std_right_box[:,0] - np.sin(theta)*self.std_right_box[:,1]
		self.right_pts[:,1] = np.sin(theta)*self.std_right_box[:,0] + np.cos(theta)*self.std_right_box[:,1]

		pos_arr = np.tile(pos, (num_pts,1))

		self.left_pts = self.left_pts + pos_arr
		self.right_pts = self.right_pts + pos_arr


	def _get_left_right_odors(self, odor_frame = None):

		self.left_odors = np.zeros(self.num_pts)
		self.right_odors = np.zeros(self.num_pts)

		for i in range(0,self.num_pts):

			if self.config['USE_MOVIE']:

				self.left_odors[i] = odor_frame[self.left_pts[i,0], self.left_pts[i,1]]
				self.right_odors[i] = odor_frame[self.right_pts[i,0], self.right_pts[i,1]]

		self.mean_left_odor = np.mean(self.left_odors)
		self.mean_right_odor = np.mean(self.right_odors)


	def update(self, theta, pos, odor_frame = None):

		self._rotate_and_translate_sensors()
		self._get_left_right_odors()
		self.concentration = 1/2(self.mean_left_odor+self.mean_right_odor)
		self.hrc = self.left_odor_prev*self.mean_right_odor - self.right_odor_prev*self.mean_left_odor
		self.gradient = self.mean_left_odor - self.mean_right_odor

		#DO TEMPORAL ONES

		self.left_odor_prev = self.mean_left_odor
		self.right_odor_prev = self.mean_right_odor
		self.odor_prev = self.concentration

		#RETURN 













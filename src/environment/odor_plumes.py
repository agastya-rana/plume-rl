import cv2
import numpy as np

class OdorPlumeFromMovie:
    def __init__(self, config, load=True):
        self.load = load
        plume_dict = config['plume']
        self.movie_path = plume_dict['MOVIE_PATH']
        self.reset_frame_range = plume_dict['RESET_FRAME_RANGE']
        self.start_frame = plume_dict['START_FRAME'] if 'START_FRAME' in plume_dict else 0
        self.stop_frame = plume_dict['STOP_FRAME']
        self.px_threshold = plume_dict['PX_THRESHOLD'] if 'PX_THRESHOLD' in plume_dict else 0
        self.mm_per_px = plume_dict['MM_PER_PX']
        self.max_conc = plume_dict['MAX_CONCENTRATION'] if 'MAX_CONCENTRATION' in plume_dict else 255
        self.video_capture = cv2.VideoCapture(self.movie_path)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS) ## Gets the FPS
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame) ## Sets the frame number to the frame number of the first frame
        self.source_location: np.ndarray = plume_dict['SOURCE_LOCATION_MM']
        self.flip: bool = False
        if load:
            _, frame = self.video_capture.read()
            frame = frame[:,:,2].T
            frame_shape = np.shape(frame)
            self.loaded_movie = np.zeros((frame_shape[0], frame_shape[1], self.stop_frame-self.start_frame))
            ## Load movie into memory
            for i in range(self.stop_frame-self.start_frame):
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame+i)
                self.loaded_movie[:,:,i] = self.read_frame()

    def reset(self, rng, flip: bool = False):
        self.frame_number = rng.integers(low=self.reset_frame_range[0], high=self.reset_frame_range[1],size=1).item()
        self.flip = flip
        self.frame = self.loaded_movie[:,:,self.frame_number-self.start_frame]

        if self.flip:

            self.frame = np.flip(self.frame, axis = 1)

    def advance(self, rng):
        if self.load:
            self.frame = self.loaded_movie[:,:,self.frame_number - self.start_frame]
            if self.flip:
                self.frame = np.flip(self.frame, axis=1)
        else:
            self.frame = self.read_frame()
            if self.flip:
                self.frame = np.flip(self.frame, axis=1)
        self.frame_number += 1

    def get_previous_frame(self):
        frame = self.loaded_movie[:,:,self.frame_number - self.start_frame - 1]
        return frame

    def read_frame(self):
        _, frame = self.video_capture.read()
        ## 2 is the red channel
        ## Transpose the frame so that the first axis is the longer axis, and the second axis is the y axis
        frame = frame[:, :, 2].T
        frame[frame<self.px_threshold] = 0 #because off plume seems to all be 1 and we want it to be 0
        frame = frame/self.max_conc
        return frame
    
    def pick_random_frame(self, rng, delay=10):
        self.frame_number = np.random.randint(self.start_frame, self.stop_frame-delay-1)
        print(self.frame_number, flush=True)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        self.advance(rng)

    
    def nearest_odor_location(self, pos):
        ## Given pos = (x, y) in mm. Returns the nearest odor location (point in frame greater than threshold) in mm
        ## Convert pos to px
        pos_px = (pos/self.mm_per_px).astype(int)
        ## Get odor locations in (x,y) px format
        odor_locations = np.argwhere(self.frame > self.px_threshold) ## Returns a list of (x,y) px coordinates np array of shape (n,2)
        ## Get the nearest odor location
        nearest_odor_location_px = odor_locations[np.argmin(np.linalg.norm(odor_locations-pos_px, axis=1))]
        ## Convert to mm
        nearest_odor_location_mm = nearest_odor_location_px*self.mm_per_px
        return nearest_odor_location_mm

class StaticGaussianRibbon(OdorPlumeFromMovie):
    def __init__(self, config):
        plume_dict = config['plume']
        self.sigma = plume_dict['RIBBON_SPREAD_MM']
        self.frame_x = plume_dict['FRAME_X_MM']
        self.frame_y = plume_dict['FRAME_Y_MM']
        self.mm_per_px = plume_dict['MM_PER_PX']
        self.source_conc = plume_dict['MAX_CONCENTRATION']
        self.source_location: np.ndarray = plume_dict['SOURCE_LOCATION_MM']
        self.stop_frame = plume_dict['STOP_FRAME']
        self.px_threshold = plume_dict['PX_THRESHOLD']
        self.frame_number = 0
        dim_x = np.rint(self.frame_x/self.mm_per_px).astype(int)
        dim_y = np.rint(self.frame_y/self.mm_per_px).astype(int)
        y_list = self.mm_per_px*np.arange(0,dim_y)
        y_coords = np.tile(y_list, (dim_x,1))
        self.frame = self.source_conc*np.exp((-(y_coords-self.source_location[1])**2)/(2*self.sigma**2))
        self.frame[self.frame<self.px_threshold] = 0

    def advance(self, rng):

        self.frame_number +=1

        return

    def reset(self, flip, rng):

        if flip:

            self.frame = np.flip(self.frame, axis = 1) #should be symmetric and not matter but just in case of rounding asymmetries

        self.frame_number = 0

        return


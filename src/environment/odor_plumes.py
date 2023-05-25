import cv2
import numpy as np

class OdorPlumeFromMovie:
    def __init__(self, config):
        plume_dict = config['plume']
        self.movie_path = plume_dict['MOVIE_PATH']
        self.reset_frame_range = plume_dict['RESET_FRAME_RANGE']
        self.frame_number = plume_dict['MIN_FRAME']
        self.start_frame = plume_dict['MIN_FRAME']
        self.stop_frame = plume_dict['STOP_FRAME']
        self.video_capture = cv2.VideoCapture(self.movie_path)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS) ## Gets the FPS
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame) ## Sets the frame number to the frame number of the first frame
        self.frame = self.read_frame()
        self.source_location: np.ndarray = plume_dict['SOURCE_LOCATION_MM']
        self.flip: bool = False
        _, frame = self.video_capture.read()
        frame = frame[:,:,2].T
        frame_shape = np.shape(frame)
        self.loaded_movie = np.zeros((frame_shape[0], frame_shape[1], self.stop_frame-self.frame_number))
        ## Load movie into memory
        for i in range(self.stop_frame-self.start_frame):
            self.loaded_movie[:,:,i] = self.read_frame()

    def reset(self, rng, flip: bool = False):
        self.frame_number = rng.integers(low=self.reset_frame_range[0], high=self.reset_frame_range[1],size=1).item()
        self.flip = flip
        self.frame = self.loaded_movie[:,:,self.frame_number-self.start_frame]

    def advance(self, rng):

        if self.frame_number >= self.stop_frame:
            self.reset(rng)
        else:
            self.frame = self.loaded_movie[:,:,self.frame_number - self.start_frame]
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
        return frame
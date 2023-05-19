import os

#MOVIE_PATH_1 = os.path.join('src','data', 'plume_movies', 'intermittent_smoke.avi')
from typing import Protocol

import cv2
import numpy as np
#from numpy.random import default_rng
#from src.models.geometry import AngleField

class OdorPlumeFromMovie:
    def __init__(self, config):

        #assert os.path.isfile(movie_file_path), f'{movie_file_path} not found'
        self.movie_path = config['MOVIE_PATH']
        self.flip: bool = False
        self.reset_frame_range = config['RESET_FRAME_RANGE']
        self.frame_number = config['MIN_FRAME']
        self.stop_frame = config['STOP_FRAME']
        self.video_capture = cv2.VideoCapture(self.movie_path)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        self.frame = self.read_frame()
        self.source_location: np.ndarray = config['SOURCE_LOCATION_MM']
        self.flip: bool = False
        self.config = config
        #self.rng = np.random.default_rng(0) #change this

        _, frame = self.video_capture.read()
        frame = frame[:,:,2].T
        frame_shape = np.shape(frame)

        self.loaded_movie = np.zeros((frame_shape[0], frame_shape[1], self.stop_frame-self.frame_number))

        for i in range(config['MIN_FRAME'], config['STOP_FRAME']):

            #print('done frame ', i)

            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            frame = self.read_frame()
            self.loaded_movie[:,:,i-config['MIN_FRAME']] = frame

    def reset(self, rng, flip: bool = False):

        #self.frame_number = rand_gen.randint(low=self.reset_frame_range[0], high=self.reset_frame_range[1],size=1)
        self.frame_number = rng.integers(low=self.reset_frame_range[0], high=self.reset_frame_range[1],size=1).item()

        self.flip = flip
        self.frame = self.loaded_movie[:,:,self.frame_number-self.config['MIN_FRAME']]
        #self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        #self.frame = self.read_frame()

    def advance(self, rng):

        if self.frame_number >= self.stop_frame:
            self.reset(rng)
        else:
            self.frame = self.loaded_movie[:,:,self.frame_number - self.config['MIN_FRAME']]
            if self.flip:

                self.frame = np.flip(self.frame, axis=1)

            self.frame_number += 1

    def get_previous_frame(self):

        #self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number - 1)
        #frame = self.read_frame()
        #self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        frame = self.loaded_movie[:,:,self.frame_number - self.config['MIN_FRAME'] - 1]

        return frame

    def read_frame(self):
        _, frame = self.video_capture.read()
        frame = frame[:, :, 2].T
        return frame

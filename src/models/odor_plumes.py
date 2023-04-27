import os

#MOVIE_PATH_1 = os.path.join('src','data', 'plume_movies', 'intermittent_smoke.avi')
from typing import Protocol

import cv2
import numpy as np
from numpy.random import default_rng

from src.models.geometry import AngleField

###load config


###if config['plume_type'] == video:


MOVIE_PATH_1 = os.path.join('src','data', 'plume_movies', 'intermittent_smoke.avi')
PLUME_VIDEO_X_BOUNDS = np.array([0, 1500])  # From Nirag
PLUME_VIDEO_Y_BOUNDS = np.array([0, 900])  # From Nirag
RESET_FRAME_RANGE = np.array([501,801])
MIN_FRAME = 500
STOP_FRAME = 5000

###elif config['plume_type'] == packet_sim:

SOURCE_LOCATION = np.array([150,450])




#### NOTE: WIND DIRECTION AND SOURCE LOCATION SHOULD REALLY BE PROPERTIES OF THE PLUME

class OdorPlume(Protocol):
    frame_number: int
    frame: np.ndarray
    flip: bool

    def reset(self, flip: bool = False):
        pass

    def advance(self) -> list[int | np.ndarray]:
        pass


class OdorMotionPlume(Protocol):
    frame_number: int
    frame: np.ndarray
    source_location: np.ndarray

    def reset(self, flip: bool = False):
        pass

    def advance(self) -> list[int | np.ndarray]:
        pass

    def get_previous_frame(self) -> np.ndarray:
        pass


class OdorPlumeAllOnes:

    def __init__(self):
        self.frame_number: int = 0
        self.frame: np.ndarray = np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        self.source_location: np.ndarray = SOURCE_LOCATION
        self.flip: bool = False

    def reset(self, flip: bool = False):
        self.__init__()
        self.flip = flip
        return [self.frame_number, self.frame]

    def advance(self) -> list[int | np.ndarray]:
        self.frame_number += 1
        self.frame: np.ndarray = np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        return [self.frame_number, self.frame]

    @staticmethod
    def get_previous_frame() -> np.ndarray:
        return np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])


class OdorPlumeAllZeros:

    def __init__(self):
        self.frame_number: int = 0
        self.frame: np.ndarray = np.zeros([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        self.source_location: np.ndarray = SOURCE_LOCATION
        self.flip : bool = False

    def reset(self, flip: bool = False):
        self.__init__()
        self.flip = flip
        return [self.frame_number, self.frame]

    def advance(self) -> list[int | np.ndarray]:
        self.frame_number += 1
        self.frame: np.ndarray = np.zeros([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        return [self.frame_number, self.frame]

    @staticmethod
    def get_previous_frame() -> np.ndarray:
        return np.zeros([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])


class OdorPlumeAlternating:
    def __init__(self):
        self.frame_number: int = 0
        self.frame: np.ndarray = np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        self.source_location: np.ndarray = SOURCE_LOCATION
        self.flip: bool = False

    def reset(self, flip: bool = False):
        self.__init__()
        self.flip = flip
        return [self.frame_number, self.frame]

    def advance(self) -> list[int | np.ndarray]:
        self.frame_number += 1
        if self.frame_number % 2 == 1:
            self.frame: np.ndarray = np.zeros([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        else:
            self.frame: np.ndarray = np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        return [self.frame_number, self.frame]

    def get_previous_frame(self) -> np.ndarray:
        prior_frame_number = self.frame_number - 1
        if prior_frame_number % 2 == 1:
            prior_frame = np.zeros([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        else:
            prior_frame = np.ones([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        return prior_frame


"""
class OdorPlumeGradientY:
    def __init__(self, slope: float = 1):
        self.frame_number: int = 0
        y_values = np.array([y * slope for y in range(PLUME_VIDEO_Y_BOUNDS[1])])
"""


class OdorPlumeRollingRandom:

    #put rand_gen back as required argument

    def __init__(self, roll_shift_size: int = 1, motion_direction: float = AngleField()):
        self.frame_number: int = 0
        self.shift_size = roll_shift_size
        rng = default_rng()
        self.frame: np.ndarray = rng.integers([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        #self.frame: np.ndarray = rand_gen.randint([PLUME_VIDEO_X_BOUNDS[1], PLUME_VIDEO_Y_BOUNDS[1]])
        self.source_location: np.ndarray = SOURCE_LOCATION
        self.flip: bool = False

    def reset(self, flip: bool = False):
        self.__init__()
        self.flip = flip
        return [self.frame_number, self.frame]

    def advance(self) -> list[int | np.ndarray]:
        self.frame_number += 1
        self.frame: np.ndarray = np.roll(self.frame, shift=self.shift_size)
        return [self.frame_number, self.frame]

    def get_previous_frame(self) -> np.ndarray:
        prior_frame = np.roll(self.frame, shift=-1 * self.shift_size)
        return prior_frame


class OdorPlumeFromMovie:
    def __init__(self, 
                 movie_file_path: str = MOVIE_PATH_1):

        assert os.path.isfile(movie_file_path), f'{movie_file_path} not found'
        self.movie_path = movie_file_path
        self.flip: bool = False
        self.reset_frame_range = RESET_FRAME_RANGE
        self.frame_number = MIN_FRAME
        self.stop_frame = STOP_FRAME
        self.video_capture = cv2.VideoCapture(self.movie_path)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        self.frame = self.read_frame()
        self.source_location: np.ndarray = SOURCE_LOCATION
        self.flip: bool = False
        self.rng = np.random.default_rng(0) #change this

        _, frame = self.video_capture.read()
        frame = frame[:,:,2].T
        frame_shape = np.shape(frame)

        self.loaded_movie = np.zeros((frame_shape[0], frame_shape[1], self.stop_frame-self.frame_number))

        for i in range(MIN_FRAME, STOP_FRAME):

            #print('done frame ', i)

            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            frame = self.read_frame()
            self.loaded_movie[:,:,i-MIN_FRAME] = frame

    def reset(self, flip: bool = False):

        #self.frame_number = rand_gen.randint(low=self.reset_frame_range[0], high=self.reset_frame_range[1],size=1)
        self.frame_number = self.rng.integers(low=self.reset_frame_range[0], high=self.reset_frame_range[1],size=1).item()

        self.flip = flip
        self.frame = self.loaded_movie[:,:,self.frame_number-MIN_FRAME]
        #self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        #self.frame = self.read_frame()

    def advance(self):

        if self.frame_number + 1 >= self.stop_frame:
            self.reset()
        else:
            self.frame_number += 1
            self.frame = self.loaded_movie[:,:,self.frame_number - MIN_FRAME]

    def get_previous_frame(self):

        #self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number - 1)
        #frame = self.read_frame()
        #self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        frame = self.loaded_movie[:,:,self.frame_number - MIN_FRAME - 1]

        return frame

    def read_frame(self):
        _, frame = self.video_capture.read()
        frame = frame[:, :, 2].T
        if self.flip:
            frame = np.flip(frame, axis=1)  # flip along the plume centerline
        return frame

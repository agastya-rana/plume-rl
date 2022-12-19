import numpy as np
import os
from src.models.motion_environment_factory import PlumeMotionNavigationEnvironmentMoviePlume1NiragRewardFactory, \
    PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardFactory
from src.models.action_definitions import WalkActionEnum

from matplotlib import pyplot as plt
import seaborn as sns


def main(movie_path=os.path.join('src','data', 'plume_movies', 'intermittent_smoke.avi'),
         q_table_path=os.path.join('trained_models', 'updated_q.npy'),
         savepath=os.path.join('result_images', 'q_walking_frames')):
    environment = PlumeMotionNavigationEnvironmentMovie1PlumeSourceRewardFactory(
        movie_file_path=movie_path).plume_environment

    reset = True

    done = False

    q_table = np.load(q_table_path)
    observation = environment.reset(options={'randomization_x_bounds': np.array([800, 1000]),
                                             'randomization_y_bounds': np.array([375, 525]),
                                             'flip': True})
    # observation = np.array([0, 0, 0])
    epsilon = 0
    while not done:
        sns.heatmap(environment.odor_plume.frame.T, vmin=0, vmax=250)
        if reset:
            reset = False
            while observation[0] == 0:
                action = environment.action_space.sample()
                observation, reward, done, odor_measures = environment.step(action)

        explore = np.random.rand() < epsilon
        if explore:
            action = environment.action_space.sample()
        else:
            action = np.argmax(q_table[tuple(observation)])

        # action = np.argmax(q_table[tuple(observation)])
        # action = environment.action_space.sample()
        action_enum = WalkActionEnum(action)
        displacement = environment.walk_functions[action_enum]
        compress_concentration, compress_gradient, compress_motion_speed = environment.odor_features.discretize_features()

        conc_text = f'conc.: {environment.odor_features.concentration}; compress: {compress_concentration}'
        grad_text = f'grad.: {environment.odor_features.gradient}; compress: {compress_gradient}'
        motion_text = f'mot.: {environment.odor_features.motion_speed}; compress: {compress_motion_speed}'
        position_text = f'pos.: {np.around(environment.fly_spatial_parameters.position, decimals=2)}'
        action_text = f'action: {str(action_enum).split(".")[1]}; displacement: {np.around(displacement, decimals=2)}'
        total_text = '\n'.join([conc_text, grad_text, motion_text, position_text, action_text])

        new_observation, reward, done, odor_measures = environment.step(action)
        plt.scatter(environment.fly_spatial_parameters.position[0], environment.fly_spatial_parameters.position[1])
        frame_num_str = (str(environment.odor_plume.frame_number).zfill(4))
        frame_fn = frame_num_str + '.png'
        frame_path = os.path.join(savepath, frame_fn)

        plt.gca().text(0, 200, total_text, color='white')
        plt.savefig(frame_path)
        plt.close(plt.gcf())
        observation = new_observation

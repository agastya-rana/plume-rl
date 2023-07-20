import os
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

class LogSuccessSaveModel(BaseCallback):

	def __init__(self, config, verbose=True):
		super().__init__(verbose)
		self.save_freq = config["training"]["SAVE_FREQ"] ## around 5M steps
		self.log_freq = config["training"]["LOG_FREQ"] ## around 400K steps
		self.success_history = []
		self.save_dir = os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME'])
		self.init_flag = False

	def _on_step(self):
		if not self.init_flag:
			self.init_flag = True
			if isinstance(self.training_env, VecEnv):
				self.n_envs = self.training_env.num_envs
				self.success_last_idx = [0]*self.n_envs
			else:
				self.success_last_idx = 0
		if self.n_calls % self.save_freq == 0:
			## Save the model
			self.model.save(self.save_dir)
			print("Model saved at step ", self.n_calls/1000000, "M")
		if self.n_calls % self.log_freq == 0:
			## Save the success percentage to TB logger
			## Check if training_env is vectorized or not
			if isinstance(self.training_env, VecEnv):
				successes = self.training_env.get_attr("all_episode_success")
				all_successes = []
				for i in range(self.n_envs):
					all_successes += successes[i][self.success_last_idx[i]:]
					self.success_last_idx[i] = len(successes[i])
				current_success = np.mean(all_successes)
			else:
				current_success = np.mean(self.training_env.all_episode_success[self.success_last_idx:])
				self.success_last_idx = len(self.training_env.all_episode_success)
			self.logger.record('success', current_success)
			## Save model if it is best
			if len(self.success_history) == 0 or current_success > np.max(self.success_history):
				self.model.save(self.save_dir+"_best")
			## Save the success percentage to success_history
			self.success_history.append(current_success)
		return True

	def _on_training_end(self):
		## Save the success history to a file
		np.save(self.save_dir+"_training_success.npy", np.array(self.success_history))
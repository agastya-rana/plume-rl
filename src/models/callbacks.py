import os
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class LogSuccessSaveModel(BaseCallback):

	def __init__(self, config, verbose=True):
		super().__init__(verbose)
		self.save_freq = config["training"]["SAVE_FREQ"] ## around 5M steps
		self.log_freq = config["training"]["LOG_FREQ"] ## around 400K steps
		self.success_history = []
		self.save_dir = os.path.join(config["training"]["SAVE_DIRECTORY"], config["training"]['MODEL_NAME'])
	
	def _on_step(self):
		if self.n_calls % self.save_freq == 0:
			## Save the model
			model.save(self.save_dir)
			print("Model saved at step ", self.n_calls/1000000, "M")
		if self.n_calls % self.log_freq == 0:
			## Save the success percentage to TB logger
			current_success = np.mean(self.success_history[-self.log_freq:])
			self.logger.record('success', current_success)
			## Save model if it is best
			if current_success > np.max(self.success_history):
				model.save(self.save_dir+"_best")
			## Save the success percentage to success_history
			self.success_history.append(current_success)
		return True

	def _on_training_end(self):
		## Save the success history to a file
		np.save(self.save_dir+"_training_success.npy", np.array(self.success_history))
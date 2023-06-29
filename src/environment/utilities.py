## Utilities used in training and testing
import pickle, os
## Function to print nested dictionary over many lines in a visually appealing way
def print_dict(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key) + ": ")
        if isinstance(value, dict):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

## Functions to store and load the config dict as pickle file
def store_config(config):
    with open(os.path.join(config["training"]["SAVE_DIRECTORY"], f'{config["training"]["MODEL_NAME"]}_config.pkl'), 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

def load_config(model_name, save_dir):
    with open(os.path.join(save_dir, f"{model_name}_config.pkl"), 'rb') as f:
        config = pickle.load(f)
    return config
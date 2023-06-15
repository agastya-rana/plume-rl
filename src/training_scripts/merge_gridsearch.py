import optuna
from optuna.storages import JournalStorage, JournalFileStorage

def merge_studies(study_name, num_studies):
    # Initialize a new study
    new_study = optuna.create_study(direction='maximize', study_name=study_name+'_merged')
    # Loop over the log files
    for i in range(num_studies):
        # Load each study
        store = JournalStorage(JournalFileStorage(f"{study_name}-journal_{i}.log"))
        study = optuna.load_study(study_name=f"rnn-reward-shaping_{i}", storage=store)
        # Loop over all trials in the study
        for trial in study.trials:
            # Check if the trial is complete
            if trial.state == optuna.trial.TrialState.COMPLETE:
                # Copy the trial's parameters and value to the new study
                new_study.add_trial(
                    optuna.trial.create_trial(
                        state=optuna.trial.TrialState.COMPLETE,
                        value=trial.value,
                        params=trial.params,
                        distributions=trial.distributions,
                        user_attrs=trial.user_attrs,
                        system_attrs=trial.system_attrs,
                        intermediate_values=trial.intermediate_values,
                        #number=trial.number + i * 100  # Add an offset to the trial number to make it unique
                    )
                )
    print('Number of finished trials: ', len(new_study.trials))
    print('Best trial:')
    trial = new_study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ', trial.params)
    return new_study

if __name__ == "__main__":
    merge_studies('rnn-reward', 31)
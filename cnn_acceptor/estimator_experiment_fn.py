import tensorflow as tf

from estimator_model_fn import model_fn
from estimator_input_fn import input_fn


def experiment_fn(run_config, params):
    """
    Create an experiment to train and evaluate the model
    """
    run_config = run_config.replace(save_checkpoints_steps=params.min_eval_frequency)
    estimator = get_estimator(run_config, params)
    experiment = tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=lambda: input_fn(True, params.train_data_file, batch_size, params.train_epochs),
            eval_input_fn=lambda: input_fn(False, params.eval_data_file, batch_size),
            eval_steps=None,
            min_eval_frequency=params.min_eval_frequency
            )
    
    return experiment


def get_estimator(run_config, params):
    """
    Create an estimator for experiment
    """
    return tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params=params
            )


def get_hooks():
    """
    Create Hooks to monitor training
    """
    tensor_to_log = {'learning_rate': 'learning_rate'}
    logging_tensor_hook = tf.train.LoggingTensorHook(
            tensors=tensor_to_log,
            every_n_iter=100
            )
    return [logging_tensor_hook]


def run_experiment(params):
    """
    Run the training experiment
    """
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=model_dir)
    
    experiment = experiment_fn(run_config, params)
    
    train_hooks = get_hooks()
    experiment.extend_train_hooks(train_hooks)
    experiment.train_and_evaluate()
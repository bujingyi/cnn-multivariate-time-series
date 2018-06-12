import tensorflow as tf

from estimator_experiment_fn import run_experiment
from estimator_input_fn import input_data
    
    
if __name__ == '__main__':
    """
    main function
    """
    # create training set
    if not os.path.exists(train_file):
        input_data(*args)
    # create validation set
    if not os.path.exists(eval_file):
        input_data(*args)
        
    params = tf.contrib.training.HParams(
    	model='cnn',
        init_learning_rate=1e-4,
        train_epochs=train_epochs,
        train_steps_per_iteration=100,
        min_eval_frequency=1000,
        data_format='channels_last',
        batch_size=batch_size,
        out_width=len(predict_y),
        train_data_file=train_npy_file,
        eval_data_file=eval_npy_file,
        architecture='resnet'
       )
    
    run_experiment(params)
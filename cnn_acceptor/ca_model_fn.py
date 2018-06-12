import tensorflow as tf
import numpy as np

import estimator_model.estimator_model


def model_fn(features, labels, mode, params):
    network = estimator_model.estimator_model(params)
    preds = network(inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    predictions = {'predicted_value': preds}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # calculate loss
    loss = tf.losses.mean_squared_error(labels, preds)        
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        # initial_learning_rate = 1e-3 * params['batch_size'] / 256
        initial_learning_rate = params.init_learning_rate
        batches_per_epoch = _TOTLE_NUM['train'] / params.batch_size
        global_step = tf.train.get_or_create_global_step()
    
        # Multiply the learning rate by 0.1 at 30, 60, 80, and 90 epochs.
        boundaries = [
            int(batches_per_epoch * epoch) for epoch in [1, 4, 8, 12]]
        values = [
            initial_learning_rate * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
        tf.identity(learning_rate, name='learning_rate')
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None
        
    mse = tf.metrics.mean_squared_error(labels, preds)    
    metrics = {'mse': mse}
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics
    )
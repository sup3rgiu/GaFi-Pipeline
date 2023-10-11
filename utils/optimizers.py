import tensorflow as tf
import tensorflow_addons as tfa

tfk = tf.keras
tfkl = tf.keras.layers

def get_optimizer(optimizer, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, momentum=0.0, nesterov=False, weight_decay=None):
    if optimizer == 'adam':
        opt = tfk.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif optimizer == 'sgd':
        opt = tfk.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov)
    elif optimizer == 'sgdw':
        #opt = tfk.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay) # 'weight_decay' exists only from TF v2.11
        opt = tfa.optimizers.SGDW(learning_rate=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    elif optimizer == 'adabelief':
        opt = tfa.optimizers.AdaBelief(lr=lr, epsilon=1e-8, rectify=False)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} currently not implemented")
    
    return opt
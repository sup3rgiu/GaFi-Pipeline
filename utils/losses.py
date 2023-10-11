import tensorflow as tf

def discriminator_loss(loss_type='hinge'):
    """Calculates losses for the discriminator."""
    def _dis_loss(fake_logits, real_logits):
        if loss_type == 'hinge':
            real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logits))
            fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))
        elif loss_type == 'non-saturating':
            real_loss = tf.reduce_mean(tf.math.softplus(-real_logits))
            fake_loss = tf.reduce_mean(tf.math.softplus(fake_logits))
        else:
            raise ValueError('Discriminator loss {} not supported'.format(loss_type))
        return real_loss, fake_loss
    return _dis_loss

def generator_loss(loss_type='hinge'):
    """Calculates losses for the generator."""
    def _gen_loss(fake_logits):
        if loss_type == 'hinge':
            fake_loss = -tf.reduce_mean(fake_logits)
        elif loss_type == 'non-saturating':
            fake_loss = tf.reduce_mean(tf.math.softplus(-fake_logits))
        else:
            raise ValueError('Generator loss {} not supported'.format(loss_type))
        return fake_loss
    return _gen_loss

def r1_gradient_penalty(discriminator, inputs, penalty_cost=1.0):
    """Calculates R1 gradient penalty for the discriminator.
       Inputs is [images, labels]
    """
    images, labels = inputs
    batch_size = tf.shape(images)[0]

    with tf.GradientTape() as tape:
        tape.watch(images)
        outputs = discriminator([images, labels], training=True)

    gradients = tape.gradient(outputs, images)
    gradients = tf.reshape(gradients, (batch_size, -1))
    penalty = tf.reduce_sum(tf.square(gradients), axis=-1)
    penalty = tf.reduce_mean(penalty) * penalty_cost
    return outputs, penalty

def probs_distance(real_output, generated_output, real_y, generated_y):
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_real = criterion(real_y, real_output)
    loss_fake = criterion(generated_y, generated_output)
    return (loss_real-loss_fake)**2

def feature_distance(real_output, generated_output, real_y, generated_y):
    ''' 
     As we target image classification problems, we compute the discrepancy between the
     real and synthetic samples of the same class only
    '''
    if tf.rank(real_y) == 2:
        real_y = tf.argmax(real_y, -1)
    if tf.rank(generated_y) == 2:
        generated_y = tf.argmax(generated_y, -1)
    
    tot = 0.0
    for i in range(10):
        real_f_class = tf.gather(real_output, tf.reshape(tf.where(real_y == i), -1))
        gen_f_class = tf.gather(generated_output, tf.reshape(tf.where(generated_y == i), -1))
        dis = tf.reduce_sum((tf.reduce_mean(real_f_class, axis=0) - tf.reduce_mean(gen_f_class, axis=0))**2)
        tot += dis
    return tot
    
def probs_and_features_distances(real_output, generated_output, real_y, generated_y):
    probs_d = probs_distance(real_output['prob'], generated_output['prob'], real_y, generated_y)
    feat_d = feature_distance(real_output['feat'], generated_output['feat'], real_y, generated_y)
    return probs_d, feat_d
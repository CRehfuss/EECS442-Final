import tensorflow as tf

def placeholders():
    images = tf.placeholder(tf.float32, shape=[None, 32*32*3])
    images = tf.reshape(images, [-1, 32, 32, 3])
    labels = tf.placeholder(tf.int64, shape=[None])
    keep_prob = tf.placeholder(tf.float32)
    return images, labels, keep_prob

def optimizer(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss)
    return train_op

def cross_entropy_loss(labels, logits):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def predictions(logits):
    return tf.argmax(logits, 1)


def accuracy(labels, logits):
    correct_prediction = tf.equal(labels, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def mean_squared_error(images, reconstructed):
    mse = tf.reduce_mean(tf.squared_difference(images, reconstructed))
    return mse

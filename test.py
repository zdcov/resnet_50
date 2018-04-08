import tensorflow as tf
def weight_variable(shape):
    initial=tf.contrib.layers.xavier_initializer()
    return tf.get_variable(None,shape,tf.float32,initial)

print(weight_variable([1,3,3,1]))
from utils import *
import tensorflow as tf

model_save_path='./model_saving/v3/model'
def con2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# def weight_variable(shape):
#     """weight_variable generates a weight variable of a given shape."""
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)

def weight_variable(name,shape):
    with tf.variable_scope(name):
        w=tf.get_variable(name='w',shape=shape,initializer=tf.glorot_uniform_initializer())
        return w

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



def cost(logits,labels):
    with tf.name_scope('loss'):
        cross_entropy=tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)
    cross_entropy_cost=tf.reduce_mean(cross_entropy)
    return cross_entropy_cost

def accuracy(logits,labels):
    with tf.name_scope('accuracy'):
        correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy_op = tf.reduce_mean(correct_prediction)
    return accuracy_op

def identity_block(X_input,kernel_size,in_filter,out_filters,stage,block,training):
    """

    :param X_input: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param kernel_size:integer, specifying the shape of the middle CONV's window for the main path
    :param in_filter:python list of integers, defining the number of filters in the CONV layers of the main path
    :param out_filters:
    :param stage:integer, used to name the layers, depending on their position in the network
    :param block:string/character, used to name the layers, depending on their position in the network
    :param training:train or test
    :return:output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    block_name='res'+str(stage)+block
    f1,f2,f3=out_filters
    with tf.variable_scope(block_name):
        X_shortcut=X_input

        #first
        W_conv1=weight_variable('w_conv1',[1,1,in_filter,f1])
        x=tf.nn.conv2d(X_input,W_conv1,strides=[1,1,1,1],padding='SAME')
        x=tf.layers.batch_normalization(x,axis=3,training=training)
        x=tf.nn.relu(x)

        #second
        W_conv2=weight_variable('w_conv2',[kernel_size,kernel_size,f1,f2])
        x=tf.nn.conv2d(x,W_conv2,strides=[1,1,1,1],padding='SAME')
        x=tf.layers.batch_normalization(x,axis=3,training=training)
        x=tf.nn.relu(x)

        #third
        W_conv3=weight_variable('w_conv3',[1,1,f2,f3])
        x=tf.nn.conv2d(x,W_conv3,strides=[1,1,1,1],padding='VALID')
        x=tf.layers.batch_normalization(x,axis=3,training=training)

        #final
        add=tf.add(x,X_shortcut)
        add_result=tf.nn.relu(add)

    return add_result

def convolutional_block(X_input,kernel_size,in_filter,out_filters,stage,block,training,stride=2):
    """
    :param X_input:
    :param kernel_size:
    :param in_filter:
    :param out_filters:
    :param stage:
    :param block:
    :param training:
    :param stride:
    :return:
    """
    block_name='res'+str(stage)+block
    with tf.variable_scope(block_name):
        f1,f2,f3=out_filters
        X_shortcut = X_input

        #first
        W_conv1=weight_variable('w_conv1',[1,1,in_filter,f1])
        x=tf.nn.conv2d(X_input,W_conv1,strides=[1,stride,stride,1],padding='VALID')
        x=tf.layers.batch_normalization(x,axis=3,training=training)
        x=tf.nn.relu(x)

        #second
        W_conv2=weight_variable('w_conv2',[kernel_size,kernel_size,f1,f2])
        x=tf.nn.conv2d(x,W_conv2,strides=[1,1,1,1],padding='SAME')
        x=tf.layers.batch_normalization(x,axis=3,training=training)
        x=tf.nn.relu(x)

        #third
        W_conv3=weight_variable('w_conv3',[1,1,f2,f3])
        x=tf.nn.conv2d(x,W_conv3,strides=[1,1,1,1],padding='VALID')
        x=tf.layers.batch_normalization(x,axis=3,training=training)

        #shortcut path
        W_shortcut=weight_variable('w_shortcut',[1,1,in_filter,f3])
        X_shortcut=tf.nn.conv2d(X_shortcut,W_shortcut,strides=[1,stride,stride,1],padding='SAME')
        X_shortcut=tf.layers.batch_normalization(X_shortcut,axis=3,training=training)

        #final
        add=tf.add(X_shortcut,x)
        add_result=tf.nn.relu(add)

    return add_result

def resnet(x_input,classes=6):
    """
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    :param x_input:
    :param classes:
    :return:
    """
    x=tf.pad(x_input,tf.constant([[0,0],[3,3],[3,3],[0,0]]),'CONSTANT')

    with tf.variable_scope('reference'):
        training=tf.placeholder(tf.bool,name='training')

        #stage1
        w_conv1=weight_variable('w_conv1',[7,7,3,64])
        x=tf.nn.conv2d(x,w_conv1,strides=[1,2,2,1],padding='VALID')
        x=tf.layers.batch_normalization(x,axis=3,training=training)
        x=tf.nn.relu(x)
        x=tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

        #stage2
        x=convolutional_block(x,3,64,[64,64,256],2,'a',training,stride=1)
        x=identity_block(x,3,256,[64,64,256],stage=2,block='b',training=training)
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

        #stage3
        x=convolutional_block(x,3,256,[128,128,512],3,'a',training)
        x=identity_block(x,3,512,[128,128,512],stage=3,block='b',training=training)
        x = identity_block(x, 3, 512, [128, 128, 512], stage=3, block='c', training=training)
        x = identity_block(x, 3, 512, [128, 128, 512], stage=3, block='d', training=training)

        # stage 4
        x = convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

        # stage 5
        x=convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
        x=identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)

        x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

        flatten=tf.layers.flatten(x)
        x=tf.layers.dense(flatten,units=50,activation=tf.nn.relu)

        with tf.name_scope('dropout'):
            keep_pro=tf.placeholder(tf.float32)
            x=tf.nn.dropout(x,keep_pro)

        logits=tf.layers.dense(x,units=6,activation=tf.nn.softmax)

        return logits,keep_pro,training

def train(X_train,Y_train):

    features=tf.placeholder(tf.float32,[None,64,64,3])
    labels=tf.placeholder(tf.int64,[None,6])

    logits, keep_prob, train_mode = resnet(features)

    cross_entropy=cost(logits,labels)

    with tf.name_scope('adam_optimizer'):
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    mini_batches=random_mini_batches(X_train,Y_train,mini_batch_size=32,seed=None)

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            X_mini_batch,Y_mini_batch=mini_batches[np.random.randint(0, len(mini_batches))]
            train_step.run(feed_dict={features: X_mini_batch, labels: Y_mini_batch, keep_prob: 0.5, train_mode: True})

            if i%20==0:
                train_cost=sess.run(cross_entropy,feed_dict={features: X_mini_batch,
                                          labels: Y_mini_batch, keep_prob: 1.0, train_mode: False})
                print('step %d, training cost %g' % (i, train_cost))

        saver.save(sess, model_save_path)


def evaluate(test_features, test_labels, name='test '):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 64, 64, 3])
    y_ = tf.placeholder(tf.int64, [None, 6])

    logits, keep_prob, train_mode =resnet(x)
    acc =accuracy(logits,y_)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_save_path)
        accu = sess.run(acc, feed_dict={x: test_features, y_: test_labels,keep_prob: 1.0, train_mode: False})
        print('%s accuracy %g' % (name, accu))
data_dir = 'F:\\DL\\resnet50\\dataset'
orig_data=load_dataset(data_dir)
X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data)
train(X_train, Y_train)
evaluate(X_test, Y_test)
evaluate(X_train, Y_train, 'training data')
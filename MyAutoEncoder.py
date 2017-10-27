# -*- coding: utf-8 -*-
"""
@author: rhg
"""

class ClassMyAutoEncoder():
    
    def __init__(self):
        print('Create MyAutoEncoderClass')
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
    
    
    def MyAutoEncoder(self, data, layer, learning_rate = 0.00001, epoch = 100000):
    
        sess = tf.Session()
    
        dim = len(data[0])
    
        # input and output
        x_data = tf.placeholder(shape=[None, dim], dtype=tf.float32)
        x_target = tf.placeholder(shape=[None, dim], dtype=tf.float32)
    
        # modeling
        hidden_weight = tf.Variable(tf.random_normal(shape=[dim, layer])) #誤差項切ってみるか。
        bias1 = tf.Variable(tf.random_normal(shape=[layer]))
        output_weight = tf.Variable(tf.random_normal(shape=[layer, dim]))
        bias2 = tf.Variable(tf.random_normal(shape=[dim]))
        
        # loss function    
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(x_data, hidden_weight), bias1))
        output = tf.nn.relu(tf.add(tf.matmul(hidden_layer, output_weight), bias2))
        loss_proc = tf.matmul(tf.square(tf.add(x_target, tf.negative(output))), tf.fill([dim, 1], 1.0))
        loss = tf.reduce_mean(loss_proc)
    
        # initialize, etc
        my_opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = my_opt.minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)
    
        # exec
        loss_vec = []
        batch_size = int( len(data) / 5)
    
        for i in range(epoch):
            rand_index = np.random.choice(len(data), size=batch_size)
            rand_x = data[rand_index]
        
            sess.run(train_step, feed_dict={x_data: rand_x, x_target: rand_x})
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, x_target: rand_x})
            loss_vec.append(temp_loss)
            
            if((i+1) % 100) == 0:
                print( 'Loss:[ ' + str(temp_loss) + ']')
    
        # plot loss
        plt.plot(loss_vec[100:])
        
        # return
        return(hidden_weight)
    
    
    

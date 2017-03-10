import tensorflow as tf
import os
import pandas as pd
import numpy as np
def load():
	feature = pd.read_csv('../temp/test_features.csv', 
			dtype=np.float)
	label = pd.read_csv('../temp/test_labels.csv', 
			dtype=np.float)
	return (feature.values, label.values)
def main():
	nFeature = 42
	nLabel = 14
	nHidden = 10
	X_data, y_data = load()
	with tf.name_scope('Input'):
		X = tf.placeholder(tf.float32, [None, nFeature])
		y_true = tf.placeholder(tf.float32, [None, nLabel])
	with tf.name_scope('Hidden'):
		hidden_weight = tf.Variable(tf.random_normal([nFeature, nHidden]))
	
		hidden_bias = tf.Variable(tf.random_normal([nHidden]))
		layer1 = tf.nn.relu(tf.matmul(X, hidden_weight) + hidden_bias)
	with tf.name_scope('Output'):
		output_weight = tf.Variable(tf.random_normal([nHidden, nLabel]))
	
		output_bias = tf.Variable(tf.random_normal([nLabel]))
	
		#layer2中如果有负数，训练出来的参数都是nan。没搞明白为什么。
		#因为参数都初始化为0
		layer2 = tf.nn.relu(tf.matmul(layer1, output_weight) + output_bias)
		#用于之后的预测
		tf.add_to_collection('predict', layer2)
	with tf.name_scope('Loss'):
		abs_sub = tf.abs(layer2 - y_true)
		abs_add = tf.abs(layer2 + y_true)
		loss_nan = tf.truediv(abs_sub, abs_add)
		
		#生成和loss_all一样大小的tensor
		zeros = tf.matmul(X, tf.zeros([nFeature, nLabel]))
		loss_all = tf.where(tf.is_nan(loss_nan), zeros, loss_nan)
		loss = tf.reduce_mean(loss_all)
		tf.summary.scalar('loss', loss)
	#tf.summary.histogram('hidden_weight', hidden_weight)
	#tf.summary.histogram('output_weight', output_weight)
	#tf.summary.histogram('layer1', layer1)
	#tf.summary.histogram('layer2', layer2)
	opt = tf.train.GradientDescentOptimizer(0.01)
	train = opt.minimize(loss)
	merge_sum = tf.summary.merge_all()
	
	with tf.Session() as sess:
		#训练过程可视化
		train_writer = tf.summary.FileWriter('../temp/tf_log', sess.graph)
		sess.run(tf.global_variables_initializer())
		for k in range(1000):
			ms, _, los = sess.run([merge_sum, train, loss], feed_dict={X:X_data, y_true:y_data})
			train_writer.add_summary(ms, k)
			if los < 0.09:
				break
		saver = tf.train.Saver()
		saver.save(sess, '../temp/nn_model')
		saver.export_meta_graph('../temp/nn_model.meta')
	
#
if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	main()
import tensorflow as tf
import pandas as pd
import numpy as np
from random import randint
import os

def select_test(n, count):
    '''从大小为n的样本中随机选择count个作为
	测试，其余的用来训练'''
    index = [False for _ in range(n)]
    p = 0
    while p < count:
        ii = randint(0, n-1)
        if index[ii]:
            continue
        index[ii] = True
        p += 1
    return index
def load():
	feature = pd.read_csv('../temp/train_features.csv', 
			dtype=np.float)
	label = pd.read_csv('../temp/train_labels.csv', 
			dtype=np.float)
	feature = feature.iloc[:, 0:14]
	label = label.iloc[:, 0:14]
	index1 = feature['day1'] > 0
	index2 = label['day1'] > 0
	for k in range(2, 15):
		index1 = index1 | feature['day%d'%k] > 0
		index2 = index2 | label['day%d'%k] > 0
	index = index1 & index2
	feature = feature[index]
	label = label[index]
	test_set = select_test(feature.shape[0], round(feature.shape[0]*0.1))
	test_feature = feature.loc[test_set, :]
	test_label = label.loc[test_set, :]
	for k in range(len(test_set)):
		test_set[k] = not test_set[k]
	feature = feature.loc[test_set, :]
	label = label.loc[test_set, :]
	return (feature.values, label.values, test_feature,
				test_label)
def load_predict_feature():
	dtype = {'sid':np.str}
	for k in range(1, 15):
		dtype['day%k'] = np.float
	odat = pd.read_csv('../temp/predict_features.csv', dtype=dtype)
	sid = odat.iloc[:, 0]
	feature = odat.iloc[:, 1:]
	return (sid, feature.values)
def main():
	nFeature = 14
	nLabel = 14
	nHidden = 20
	with tf.name_scope('Input'):
		X = tf.placeholder(tf.float32, [None, nFeature])
		y = tf.placeholder(tf.float32, [None, nLabel])
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
	with tf.name_scope('Loss'):
		abs_sub = tf.abs(layer2 - y)
		abs_add = tf.abs(layer2 + y)
		loss_nan = tf.truediv(abs_sub, abs_add)
		
		#生成和loss_all一样大小的tensor
		zeros = tf.matmul(X, tf.zeros([nFeature, nLabel]))
		loss_all = tf.where(tf.is_nan(loss_nan), zeros, loss_nan)
		loss = tf.reduce_mean(loss_all)
		tf.summary.scalar('loss', loss)
	opt = tf.train.GradientDescentOptimizer(0.01)
	train = opt.minimize(loss)
	merge_sum = tf.summary.merge_all()
	
	#预测结果
	if os.path.exists('../temp/tf_model/checkpoint'):
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess, tf.train.latest_checkpoint('../temp/tf_model'))
			sid, X_data = load_predict_feature()
			#如果run的操作不依赖某个placeholder的话，可以不送数据
			y_predict = sess.run(layer2, feed_dict={X:X_data})
			data = pd.DataFrame(y_predict, columns=['day%d'%k for k in range(1, 15)])
			data.insert(0, 'sid', sid)
			data.to_csv('../temp/tensorflow_result.csv', 
				index=False, header=False, encoding='utf-8', float_format='%0.0f')
			return
	#训练
	X_data, y_data, X_test, y_test = load()
	with tf.Session() as sess:
		#训练过程可视化
		train_writer = tf.summary.FileWriter('../temp/tf_log', sess.graph)
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		for k in range(2000):
			ms, _, los = sess.run([merge_sum, train, loss], feed_dict={X:X_data, y:y_data})
			train_writer.add_summary(ms, k)
			if k % 1000 == 0:
				saver.save(sess, '../temp/tf_model/nn_model', global_step=k)
			print('%d/%d: %f' % (k, 10000, los))
			if los < 0.09:
				break 
		los = sess.run(loss, feed_dict={X:X_test, y:y_test})
		print('测试结果: %f.'%los)
		saver.save(sess, '../temp/tf_model/nn_model', global_step=10100)
#
if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	main()
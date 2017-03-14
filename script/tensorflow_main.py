import tensorflow as tf
import pandas as pd
import numpy as np
from random import randint
import os
class DataBatch:
	'''把数据分成多个batches，每次取一个batch来训练网络'''
	def __init__(self, X, y, size=1000):
		self.X = X
		self.y = y
		self.index = 0
		self.size = size
	def next_batch(self):
		if self.index + self.size <= self.X.shape[0]:
			s = self.index
			e = self.index + self.size
			self.index = e
			return (self.X[s:e, :], self.y[s:e, :])
		else:
			s = self.index
			self.index = 0
			return (self.X[s:, :], self.y[s:, :])
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
	'''加载数据，并将数据分为训练集合测试集'''
	feature = pd.read_csv('../temp/train_features.csv', 
			dtype=np.float)
	label = pd.read_csv('../temp/train_labels.csv', 
			dtype=np.float)
	#只取了前14天的销售量这一个特征
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
	'''加载待预测数据，也只取了前14天的销售量'''
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
	nHidden = 10
	#基本思路是每一天的销售量用一个3层的神经网络预测
	#并分别进行优化，然后在用官方给的Loss函数做个整体的优化
	with tf.name_scope('Input'):
		X = tf.placeholder(tf.float32, [None, nFeature])
		y = tf.placeholder(tf.float32, [None, nLabel])
	with tf.name_scope('Hidden'):
		hidden_weights = []
		hidden_biases = []
		layer1s = []
		for k in range(nLabel):
			hidden_weights.append(
				tf.Variable(tf.random_normal([nFeature, nHidden]), 
					name='hidden_weight%d'%k))
			hidden_biases.append(
				tf.Variable(tf.random_normal([nHidden]),
					name='hidden_bias%d'%k))
			layer1s.append(tf.nn.relu(tf.matmul(X, hidden_weights[k]) + hidden_biases[k]))
	with tf.name_scope('Output'):
		output_weights = []
		output_biases = []
		layer2s = []
		for k in range(nLabel):
			output_weights.append(
				tf.Variable(tf.random_normal([nHidden, 1]), 
					name='output_weights%d'%k))
			output_biases.append(
				tf.Variable(tf.random_normal([1]), 
					name='output_biases%d'%k))
			layer2s.append(
				tf.nn.relu(tf.matmul(layer1s[k], output_weights[k]) + output_biases[k]))
	with tf.name_scope('Loss'):
		sep_loss = []
		zeros1 = tf.squeeze(tf.matmul(X, tf.zeros([nFeature, 1])))
		for k in range(nLabel) :
			#对每天的预测进行评价，后面会优化
			sq = tf.squeeze(layer2s[k])
			su = tf.abs(sq - y[:,k])
			ad = tf.abs(sq + y[:,k])
			div_nan = tf.truediv(su, ad)
			div_all = tf.where(tf.is_nan(div_nan), zeros1, div_nan)
			sep_loss.append(tf.reduce_mean(div_all))
			tf.summary.scalar('loss_%d'%k, sep_loss[k])
		#整体优化
		layer2_all = tf.concat(layer2s, 1)
		abs_sub = tf.abs(layer2_all - y)
		abs_add = tf.abs(layer2_all + y)
		loss_nan = tf.truediv(abs_sub, abs_add)
		
		#生成和loss_all一样大小的tensor
		zeros = tf.matmul(X, tf.zeros([nFeature, nLabel]))
		loss_all = tf.where(tf.is_nan(loss_nan), zeros, loss_nan)
		loss = tf.reduce_mean(loss_all)
		tf.summary.scalar('loss', loss)
	opt = tf.train.GradientDescentOptimizer(0.01)
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
	bch = DataBatch(X_data, y_data)
	print('Data loaded!')
	with tf.Session() as sess:
		#训练过程可视化
		train_writer = tf.summary.FileWriter('../temp/tf_log', sess.graph)
		sess.run(tf.global_variables_initializer())
		#print(sess.run([su[0], ad[0],sep_loss[0]], feed_dict={X:X_data, y:y_data}))
		saver = tf.train.Saver()
		#分别训练
		for k in range(10):
			bx, by = bch.next_batch()
			for e in range(14):
				train = opt.minimize(sep_loss[e])
				ms, _, los = sess.run([merge_sum, train, sep_loss[e]], 
							feed_dict={X:bx, y:by})
				print('%d - day%d: %f.'%(k, e, los))
			train_writer.add_summary(ms, k)
		#整体优化
		for k in range(10):
			train = opt.minimize(loss)
			bx, by = bch.next_batch()
			ms, _, los = sess.run([merge_sum, train, loss], 
						feed_dict={X:bx, y:by})
			train_writer.add_summary(ms, k+10)
			if k % 100 == 0:
				saver.save(sess, '../temp/tf_model/nn_model', global_step=k)
			print('%d/%d: %f' % (k, 10, los))
		
		los = sess.run(loss, feed_dict={X:X_test, y:y_test})
		print('测试结果: %f.'%los)
		saver.save(sess, '../temp/tf_model/nn_model', global_step=10100)
#
if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	main()
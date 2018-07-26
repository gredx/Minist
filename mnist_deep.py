#!/usr/bin/env python
# load MNIST data  数据输入
import input_data
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

# start tensorflow interactiveSession
import tensorflow as tf
sess = tf.InteractiveSession()
   
# weight initialization
# 权值和偏置量初始化函数 , 传入变量大小 , 进行初始化
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

# convolution
# 卷积和池化函数 , 定义函数为了方便的调用
# 卷积传入 x 和 w , 池化 传入 x
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Create the model 	开始创建网络
# placeholder	输入x和输出y 先设置占位符
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
# variables		设置权值和偏置量为 变量,并指定空间大小
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 建立 y 的图...
y = tf.nn.softmax(tf.matmul(x,W) + b)

# first convolutinal layer
# 第一个卷积层
# 卷积核的大小是5*5 , 输入是灰度图,所以输入通道是1 , 输出通道设为 32 , 偏置项的大小要和输出通道一致
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])	# 输入x 是28*28 的图 , 是灰度图 , -1 代表数目不定

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)	# 将输入卷积后 使用 relu 激活函数 得到输出
h_pool1 = max_pool_2x2(h_conv1)			#　池化处理后输出

# second convolutional layer
# 第二个卷积层
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
# 一个全连接层
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")		# 设置屏蔽神经元的概率 , 一般是0.5
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer
# 输出层
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# train and evaluate the model
# 训练和评估 部分
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)	# 做最速梯度下降 , 反向传播
# 一次train_step 就是一次训练
#train_step = tf.train.AdagradOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 是否是 一次正确的预测
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))		# 计算精确度
sess.run(tf.initialize_all_variables())
# 进行训练
for i in range(20000):
	# 一次传入50组数据
	batch = mnist.train.next_batch(50)
	# 每隔100次 输出一次精确度
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
		print("step %d, train accuracy %g" %(i, train_accuracy))
	train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

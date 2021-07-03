简单的hello world。使用纯粹的Tensorflow来识别MNIST手写库，不使用中间层，直接使用softmax进行输出，最终准确率为90%左右。

使用Tensorflow 1的语法，因此调用`disable_v2_behavior`强行禁用v2行为。


get_batch：取出一个batch。 因为tensorflow.examples.tutorials已经被移动到keras里面，所以这里直接手动用slice来模拟batch。

load_data，使用np.load加载npz文件。mnist.npz是zip格式的，里面带四个文件x_test,x_train，y_test,y_train。也正好对应四个数据集。



#数据准备



载入后，对载入的训练集和测试集都/255.0来进行归一化，方便后续运算。

但通过np.load加载的图像数据（x）会变成三维数组，也就是?, 28, 28，为了后面统一计算，将其像素部分合并，改为?, 784的数据。

另外，y也需要处理，直接加载进来的y就是对应数字的值，比如7这样。而我们使用softmax，这需要y对应所有可能的输出的概率，例如7应该对应[0,0,0,0,0,0,0,1,0,0]。所以还需要对y进行转换。

使用one_hot可以很好的解决这个需求，(同时交叉熵ylogp也需要y是onehot编码)，因为数字原本的类型就是整数，这里通过onehot转换到depth=10会得到一个长度为10的一维张量，符合我们的需求。

##模型准备##
这次不使用中间层，输入是一个784x1的tensor，每个像素点xi对应一个权重wi，加上偏置项b，然后通过softmax输出10个类的概率。损失函数为交叉熵，这是分类中比较常见的损失函数。

所以实际上是在做：
   cross_entropy(softmax(x * w + b), y)
当预测值与标签足够接近时，交叉熵会变小。采用梯度下降法对交叉熵的缩减。
学习率固定，设为0.5； 轮数300； 批处理大小100。



#算法实现



首先使用
`x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])`
定义一个占位，形状为?, 784，因为我们这里假定不知道有多少输入数据。
同样的，对y'（实际分布，即mnist里面的标签），也做一个占位
`y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])`

根据规划，784个像素，每个像素都有一个w，最终输出10类，所以w应当有形状[784, 10]，这里先全部初始化为0。
`w = tf.Variable(tf.zeros([784, 10]))`

在w * x之后，还要加上偏置项。因为输出是10类，所以这里加法必然也是个10类的矩阵，不然形状对不上了：
`b = tf.Variable(tf.zeros([10]))`

然后，写出预测分布（即预测出的结果）的softmax的实现：
`y = tf.nn.softmax(tf.matmul(x, w) + b)`

定义交叉熵H=-Σy'logy。
`cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))`
以及训练用的对象，设置learning rate=0.5，目标是最小化交叉熵。
`train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)`



#开始训练



只有在session中才能运行优化步骤train_step。而之后的sess.run(tf.global_variables_initializer())  则是为了初始化里面的所有变量以及分配内存用的。所以无论如何，基本每个工程都是要执行这两句的：
`sess = tf.InteractiveSession()
tf.global_variables_initializer().run()`

tf.InteractiveSession()可以先构建一个session然后再定义操作。
但如果使用tf.Session()，则需要在会话构建之前就定义好全部的操作再构建会话。

然后，在循环中，每次抽取100个样本，送给train_step。feed_dict的作用是给使用placeholder创建出来的tensor赋值。
`sess.run(train_step, feed_dict={x: xb, y_: yb})`


接下来，定义准确度度量函数， accuracy = 1/n Σ sgn(y, y')

softmax为预测值生成了10个不同的可能性，可能性最高的那个就是预测结果，这里通过tf.argmax(y, 1)，1是axis来找到这个结果。
例如
`[0, 0.1, 0.3, 0.4, 0.2]
  0   1    2    3    4`
则输出为3，因为3符合argmax。

而实际标签值最高的就是标签所在的那个index（值为1.0），通过tf.equal实现了sign函数。而接着它将所有值汇总取平均数，如果100%预测正确，那必然平均数就等于样本数，而实际肯定是不到的，因此通过这个就能看到预测正确的比例：

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

并根据它来评估每一轮训练集的准确度。
 `   accuracy.eval(test_data)`



#结论



这是我使用纯Tensorflow去编写学习模型的第一个用例，因此整体以简单为主，方便学习各函数的用法。

而在一定的epoch之后，它的准确率便不再变化了，因此可以提前结束学习。另外，这里使用了固定学习率，这也会导致效果不佳，下一个工程我会尝试在输入和输出中间添加一层隐藏层，并动态调整学习率。



> epoches 298, and its accuracy is: 0.903700
> epoches 299, and its accuracy is: 0.905700
> epoches 300, and its accuracy is: 0.905700
> epoches 301, and its accuracy is: 0.905700
> epoches 302, and its accuracy is: 0.905700


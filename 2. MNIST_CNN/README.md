首先，工程大体结构和之前的类似。参考了多个别人写的工程，按自己的理解修改了一些内容。这次使用CNN，CNN有很多变体，但大多基于层级模式，分为

```
  []    | 卷积,ReLu,池化,卷积,ReLu,池化……  |  全连接层
--------+--------------------------------+------------
输入层  |            特征提取层            |   分类层
```

因此这次定义的层级结构如下，即一个非常有代表性的卷积神经网络：

```
输入 --> 卷积层1 --> 池化1 -->  卷积层2 --> 池化2 --> 全连接层1 --> [dropout] --> 全连接层2 
```

之所以使用2个全连接层，可能是因为：两层的全连接网络理论上就可以拟合任何连续可微的函数，或者：

> https://www.quora.com/What-is-the-purpose-of-using-more-than-1-fully-connected-layer-in-a-convolutional-neural-network
>
> I think the reason is that the conv layers extract high level features and the fully connected layers decide the non-linear function from these features. You just use more than 1 layer to make it non-linear.



其中各层定义如下：

输入层
-----
首先，输入层数据由下列Placeholder定义：

```
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
```

而因为CNN的输入层的输入想要保留图片本身的结构。所以对MNIST而言，这里不能直接丢784的一维数据进去，而应该转为28 像素 x 28 像素 x 1个颜色通道的形式。也即shape为 ? x 28 x 28 x 1。

不过和placeholder的shape不同，这里似乎**不能**用None来指定不确定的大小，官方手册只提到使用-1。

```
https://www.tensorflow.org/api_docs/python/tf/reshape
If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. In particular, a shape of [-1] flattens into 1-D. **At most one** component of shape can be -1.
```

因此，设置x_image为x转为[-1, 28, 28, 1] shape后的值。

```
x_image = tf.reshape(x, [-1, 28, 28, 1])
```




卷积层1和卷积层2
----- 

根据资料有：

> 每一层卷积有多少channel数，以及一共有多少层卷积，这些暂时没有理论支撑，一般都是靠感觉去设置几组候选值，然后通过实验挑选出其中的最佳值。

> 大部分卷积神经网络都会采用逐层递增（1⇒ 3 ⇒ 5 ⇒ 7）的方式。而每经过一次池化层，卷积层过滤器的深度都会乘以 2。

> 对于卷积核数你可以尝试一下两倍两倍地增加，推荐用小的卷积核搭配大量的卷积核数，可以提高网络的分辨能力，理论上当然越多越好。

>但是过多的参数也会导致过拟合问题，所以在试的时候，你可以把loss打印出来观察其变化情况，如果loss后期震荡不下降，而验证集的loss在增加那么可能过拟合了。这样子做几个实验基本上就能确定卷积核数量的范围了。

设置卷积核为1x1可以增加网络的非线性，使得网络可以表达更加复杂的特征。但不是我这个问题关注的内容。

因此为了保证感受野能看到足够大的区域，这里我就按经验值使用了 3x3  -> 5x5 的模式（也试了下设置5x5 -> 5x5，没有什么显著区别）。

即第一层设置了3x3的卷积核，个数是32个。而下一层则使用5x5的卷积核，个数也加倍，64个。

第一层，对应卷积核的权重形状为[3,3,1,32]，偏置项大小即卷积核大小[32]。通常连续分布的数据使用ReLU机或函数进行建模，所以这里也套入relu。然后，加入池化层。

看网上的实现，大部分代码都加上了这个：truncated_normal。 truncated_normal用来生成截断正态分布，范围 [ mean - 2 * stddev, mean + 2 * stddev ]。目的是保证初始化的参数中没有比较大的数。因为比较大的数字意味着更大的抖动，更大的抖动带来更大的梯度，容易导致模型不收敛。

```
w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
```



池化层1和池化层2
-----
池化层夹在连续的卷积层中间， 用于压缩数据和参数的量，减小过拟合。如果输入是图像的话，那么池化层的最主要作用就是压缩图像。
>池化层的主要作用有三个：1. 增加CNN特征的平移不变性。2. 池化的降采样使得高层特征具有更大的感受野。 3. 池化的逐点操作相比卷积层的加权和更有利于优化求解。

>https://www.cnblogs.com/eilearn/p/9282902.html

tf.nn.max_pool(value, ksize, strides, padding, name=None)参数和卷积很类似：

value：需要池化的输入，通常是feature map。
ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]。 因为我们不想在batch和channels上做池化， 所以这两个维度设为了1。
strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]。
padding：和卷积类似，可以取'VALID' 或者'SAME'。

>https://blog.csdn.net/xgyyxs/article/details/102640752
strides即代表在四个维度（batch、 height 、width、channels）所移动的步长。
ksize即代表在四个维度（batch、 height 、width、channels）池化的尺寸。

最常用的下采样是取最大值的max操作（其次是平均池化），池化层最常见的设置是应用2x2的过滤器，步长为2。在池化过程中，大量应用到SAME填充。
因此，按常见设置，选择卷积核为2x2的（max）池化层，步长为2，即：

```
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

并为池化层2配置相同的参数。然后，计算各层的大小。

对卷积层1而言，输入是28x28x1，其卷积层kernel为3x3，步长为1，因此padding后等价于一个30x30的图，卷积后得到28x28  的形状，加上32个卷积核，形状为28x28x32，对比输出的形状。

```
              26  27  28
[] [] [] .... [] [x] [x] [x]
              [] [x] [x] [x]
              [] [x] [x] [x]
                          ^padding
              26 27 28
[] [] [] .... [] [] [x] [x] [x]
              [] [] [x] [x] [x]
              [] [] [x] [x] [x]
                          ^  ^ padding  (same)
```

Same和valid的区别：						  
https://www.cnblogs.com/willnote/p/6746668.html


对池化层1而言，输入是28x28x32，其卷积层kernel为2x2，步长为2，因此得到14x14 （28/2）x（28/2）的形状，即14x14x32。

对卷积层2而言，输入是14x14x32，其卷积层kernel为5x5，步长为1，padding后等价于一个18x18的图，卷积后得到14x14 (18-5+1)x(18-5+1)的形状，64个卷积，即14x14x64。

对池化层2而言，输入是14x14x64，其卷积核为2x2，步长为2，得到7x7x64的输出。实际使用时，可以直接np.shape(....)来得到结果，不用手算。

所以，前几层的状态如下：

```
类型     | Kernel尺寸 | 步长  | 输出尺寸
--------+------------+------+----------
输入层   |                   | ?x 28x28x1
卷积层1  | 3x3x32     |  1   | ?x 28x28x32
池化层1  | 2x2        |  2   | ?x 14x14x32
卷积层2  | 5x5x64     |  1   | ?x 14x14x64
池化层2  | 2x2        |  2   | ?x 7x7x64
```

全连接层1
-----
全连接层接收池化层2的输入是?x 7x7x64，因此这里也应该填入7x7x64。至于输出层的大小1024，则纯粹是因为大家都这么写，所以这里也写1024。（难道是因为MNIST有10个可能，取了个2^10凑整？……

>CNN卷积神经网络的全连接层为什么要有一层1024神经元？
https://www.zhihu.com/question/313021600

> Flatten层  https://blog.csdn.net/qq_28835913/article/details/85335369
　　截止到Flatten层之前，在网络中流动的数据还是多维的（对于我们的程序就是2维的），经过多次的卷积、池化、Dropout之后，到了这里就可以进入全连接层做最后的处理了。全连接层要求输入的数据必须是一维的，因此，我们必须把输入数据“压扁”成一维后才能进入全连接层，Flatten层的作用即在于此。该层的作用如此纯粹，因此反映到代码上我们看到它不需要任何输入参数。 
  
在池化和全连接层中间还有一个flatten的操作，实际上就是将数据压成一维的。因为池化层2输出是?x 7x7x64，有n个图片，因此这里将池化层2的输出压缩为 ? x (7*7*64)的。在经过relu(flat x w_fc1 + b_fc1)之后，输出的形状是?x 1024。

```
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
```

dropout
------
为了提高准确度，在最终的全连接层前面加入dropout，训练时丢弃50%数据，测试时不丢弃数据。dropout层相对简单，输出大小不变

```
keep_prob = tf.compat.v1.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

全连接层2
-----
因为上一层传来的是[1024]，因此这层形状为[1024, 10]以输出[10]形状的预测。

```
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
```

因此，各层参数如下：

```
类型     | Kernel尺寸 | 步长  | 输出尺寸
--------+------------+------+----------
输入层   |                   | ?x 28x28x1
卷积层1  | 3x3x32     |  1   | ?x 28x28x32
池化层1  | 2x2        |  2   | ?x 14x14x32
卷积层2  | 5x5x64     |  1   | ?x 14x14x64
池化层2  | 2x2        |  2   | ?x 7x7x64
全连接层1|                   | ?x 1024
Dropout |                   | ?x 1024
全连接层2|      分类输出       | ?x 10
```



损失函数
-----
因为仍然是分类问题，因此使用交叉熵作为优化目标，即用预测出的内容与原始标签计算交叉熵。

> 如何选择优化器呢？  https://blog.csdn.net/qq_32172681/article/details/100979476
如果数据是稀疏的，就用自适用方法，即 Adagrad, Adadelta, RMSprop, Adam。
RMSprop, Adadelta, Adam 在很多情况下的效果是相似的。
Adam 就是在 RMSprop 的基础上加了 bias-correction 和 momentum，随着梯度变的稀疏，Adam 比 RMSprop 效果会好。
整体来讲，Adam 是最好的选择。
很多论文里都会用 SGD，没有 momentum 等。SGD 虽然能达到极小值，但是比其它算法用的时间长，而且可能会被困在鞍点。
如果需要更快的收敛，或者是训练更深更复杂的神经网络，需要用一种自适应的算法。

因此这里优化器则选择步长为0.001的Adam。
定义为：

```
tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam'
)
```

```
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
```


训练
------
训练部分的代码和第一次的差不多。就不重复写了。使用上述方案训练后，准确度明显优于无连接的版本。但仍然没有达到非常高的地步。 

同时，这次训练已经明显感受到CPU版本的tensorflow运行速度之慢了，而且dropout还加剧了这个慢速的过程。

配置pycharm使用anaconda环境，anaconda中安装tensorflow-gpu后，速度大幅提升。

```
step 320, acc: 0.9623
step 340, acc: 0.9572
step 360, acc: 0.9566
step 380, acc: 0.9564
step 400, acc: 0.9564
step 420, acc: 0.9564
```


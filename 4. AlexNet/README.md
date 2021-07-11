AlexNet典型的案例是参加ILSVRC 2012（ImageNet Large Scale Visual Recognition Challenge）比赛时用的子数据集。但它的内容也十分巨大，有154GB，1000个分类。
根据https://www.tensorflow.org/datasets/catalog/imagenet2012_subset的指引去官网注册了个账号，然而他们并不提供下载给gmail之类的域名，后去Academic Torrents搜索发现了可下载的种子但是又是无标签的版本。

考虑到我只是本地测试，而且只有一张RTX 1060x6GB的显卡，而即使是降采样的版本，运行速度也难以忍受。为了快点先过完基本的网络结构，这里我将使用CIFAR-100来替代它，暂且用这个类别少点的CIFAR-100模拟ILSVRC 2012 ImageNet吧。


AlexNet
==========
AlexNet参赛时使用的网络配置如下：

输入 227x227x3 -> 
卷积层 11x11, pad = 4, 96 kernels -> 
最大池化层 3x3, strides = 2 -> 
卷积层 5x5, pad = 2, 256 kernels ->
最大池化层 3x3, strides = 2 -> 
卷积层 3x3, pad = 1, 384 kernels ->
卷积层 3x3, pad = 1, 384 kernels ->
卷积层 3x3, pad = 1, 256 kernels ->
最大池化层 3x3, strides = 2 ->
全连接层 4096 ->
全连接层 4096 ->
softmax  1000


中间的pad（0填充边缘）作用是让输入图像不变小，而可以使用更深层的卷积。


在CIFAR-10上适用AlexNet
==========
（2021.7.10）因为数据下载时间有点长，在这个期间内，我先使用tf.keras依葫芦画瓢，照着论文里面的结构做了一个alexnet。比较惨的是代码忘记移出来了，已经被我覆盖掉，不过代码结构和CIFAR-100一样，可以直接看CIFAR-100的代码。

直接应用上述“网络使用”节里面的网络，梯度下降使用Adam（而不是论文中提到的sgd），且不加入Dropout，迭代过程中便出现了过拟合的现象。
Epoch 20/20
782/782 [==============================] - 31s 39ms/step - loss: 0.5243 - accuracy: 0.8158 - val_loss: 0.7617 - val_accuracy: 0.7490

在全连接层和softmax中间加入0.25的droupout层后重新训练。过拟合现象消除，测试准确率大约在75%左右。
Epoch 20/20
782/782 [==============================] - 31s 40ms/step - loss: 0.5358 - accuracy: 0.8119 - val_loss: 0.7759 - val_accuracy: 0.7523


在全连接层和softmax中间加入0.25的dropout层，且使用sgd进行梯度下降，正如论文中提到的：

>我们使用随机梯度下降来训练我们的模型，样本的batch size为128，动量为0.9，权重衰减为0.0005。我们发现少量的权重衰减对于模型的学习是重要的。换句话说，权重衰减不仅仅是一个正则项：它减少了模型的训练误差。


opt = keras.optimizers.SGD(momentum=0.9, decay=0.0005, nesterov=False)

多次进行训练，得到结果基本如下：

Epoch 20/20
782/782 [==============================] - 31s 39ms/step - loss: 0.5828 - accuracy: 0.7954 - val_loss: 0.7238 - val_accuracy: 0.7571


最后，和论文一致，在全连接层和softmax中间加入0.50的dropout层，且使用sgd进行梯度下降：


Epoch 20/20
782/782 [==============================] - 32s 41ms/step - loss: 0.5997 - accuracy: 0.7893 - val_loss: 0.7514 - val_accuracy: 0.7490


看起来似乎并不高，不过可能是因为原来它的输入图片都很大，所以第一步就使用了11x11的卷积核，而我的输入只有32x32x3，因此如果用11x11的卷积核会显得过大。我将第一步修改为7x7和5x5的卷积核分别试验，得到下列结果：

7x7
Epoch 20/20
782/782 [==============================] - 31s 39ms/step - loss: 0.5466 - accuracy: 0.8084 - val_loss: 0.6711 - val_accuracy: 0.7777

5x5
Epoch 20/20
782/782 [==============================] - 28s 35ms/step - loss: 0.5036 - accuracy: 0.8234 - val_loss: 0.6227 - val_accuracy: 0.7931

7x7相比于11x11结果类似，但5x5的改进则很大，虽然仍有过拟合的嫌疑，但准确率数值好于11x11和7x7。


另外，keras.optimizers.SGD默认的学习率是0.01，loss没有出现明显的大幅震荡，但下降也很慢。尝试向下降低学习率则现象加剧；向上增大学习率，改为lr=0.03，大学习率在前期进展很快，几乎每个epoch都提升5%，但是后期则使得整体陷入震荡状态，反而不利于准确率的提升，在20epoch结束后训练结果为:

Epoch 20/20
782/782 [==============================] - 27s 35ms/step - loss: 0.4786 - accuracy: 0.8322 - val_loss: 0.6830 - val_accuracy: 0.7769

数字稍有提升，但感觉也有点过拟合的样子。所以实际上可以试试较大的学习率，搭配上合适的衰减速度，可能会在某些程度上提高准确率。


在CIFAR-100上适用AlexNet
==========
CIFAR-100是一个包含 50,000 张 32x32 彩色训练图像和 10,000 张测试图像的数据集，标记了 100 多个细粒度类，这些细粒度类又分为 20 个粗粒度类。

这里我用细粒度类来测试。测试的网络即上一次测试的，lr=0.03，衰减=0.0005，sgd，网络后三层有0.5的dropout。


Epoch 20/20
782/782 [==============================] - 27s 35ms/step - loss: 1.6637 - accuracy: 0.5390 - val_loss: 2.1325 - val_accuracy: 0.4608

不过应该是训练轮次的问题，在到达这个epoch之后依然没有收敛的态势，当训练次数增长时，准确率应当也会随之增长。


使用tf.layers改写AlexNet并增加训练次数
==========
（2021.7.11）总结一下之前的经验，与keras的Conv2D对应的layer为：
 tf.compat.v1.layers.conv2d() 

与MaxPooling2D对应的则只有底层api，即tf.nn系列的：
 tf.nn.max_pool() 

与Flatten对应的当然就是reshape操作：
 tf.reshape()

与Dropout对应的是也是底层的api，tf.nn系列的：
 tf.nn.dropout()
 
与Dense对应的全连接层操作是tf.layers.dense：
 tf.layers.dense()

有了上述经验，加上keras打出来的层级结构，则很容易得到对应的tf.layer的表示。

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 96)        7296      
max_pooling2d (MaxPooling2D) (None, 15, 15, 96)        0         
conv2d_1 (Conv2D)            (None, 8, 8, 256)         614656    
max_pooling2d_1 (MaxPooling2 (None, 3, 3, 256)         0         
conv2d_2 (Conv2D)            (None, 3, 3, 384)         885120    
conv2d_3 (Conv2D)            (None, 3, 3, 384)         1327488   
conv2d_4 (Conv2D)            (None, 3, 3, 256)         884992    
max_pooling2d_2 (MaxPooling2 (None, 1, 1, 256)         0         
flatten (Flatten)            (None, 256)               0         
dense (Dense)                (None, 4096)              1052672   
dropout (Dropout)            (None, 4096)              0         
dense_1 (Dense)              (None, 4096)              16781312  
dropout_1 (Dropout)          (None, 4096)              0         
dense_2 (Dense)              (None, 100)               409700    
=================================================================


前面的数据读取我就还保留keras自己的了。实现过程中出现了下列问题：
4logits and labels must be broadcastable: logits_size=[50,100] labels_size=[200,100]
	 [[node softmax_cross_entropy_with_logits_sg (defined at machine_\main.py:90) ]]

经查原来用的是valid padding，我给写成了same，这样从第二个池化层开始大小就不对了，最终计算下来导致形状错误。修正后，代码可以正常运行：

conv1, (?, 32, 32, 96)
pool1, (?, 15, 15, 96)
conv2, (?, 8, 8, 256)
pool2, (?, 3, 3, 256)
conv3, (?, 3, 3, 384)
conv4, (?, 3, 3, 384)
conv5, (?, 3, 3, 256)
pool3, (?, 1, 1, 256)
flatten, (?, 256)
fc1, (?, 4096)
fc2, (?, 4096)
fc3, (?, 100)
yconv, (?, 100)


为加快训练速度，设置dropout=0.15，使用Adam（衰减0.999）。但使用类似轮数的训练只能得到约20%的准确率，相比上一个45%的准确率只有一半，这里还需要后面再仔细进行研究。


使用tf.nn改写AlexNet并增加训练次数
==========
（2021.7.11）使用tf.nn实现则需要自己再实现最简单的层级模型，卷积层(激活函数为ReLU)可以定义为：
```
def conv_2d(input, conv_size, input_depth, output_size):
    w_conv = tf.Variable(tf.truncated_normal([conv_size, conv_size, input_depth, output_size], stddev=0.01))
    b_conv = tf.Variable(tf.constant(0.1, shape=[output_size]))
    h_conv = tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv)
    return h_conv

```

而全连接层可以定义为：
```
def max_pooling(input, kernel_size, pool_strides=2):
    h_pool = tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, pool_strides, pool_strides, 1], padding='SAME')
    return h_pool
```

而全连接层也可以定义为：
```
def dense(input, out_size, stddev, activation="relu", no_bias=False):
    w_fc = tf.Variable(tf.truncated_normal([input.shape[1], out_size], stddev=stddev))
    bias_init = 0.01
    if no_bias:
        bias_init = 0.00

    b_fc = tf.Variable(tf.constant(bias_init, shape=[out_size]))
    if activation == "relu":
        h_pool_fc = tf.nn.relu(tf.matmul(input, w_fc) + b_fc)
        print("ReLU called, shape:", np.shape(h_pool_fc))
        return h_pool_fc
    elif activation == "softmax":
        h_pool_fc = tf.nn.softmax(tf.matmul(input, w_fc) + b_fc)
        print("softmax called, shape:", np.shape(h_pool_fc))
        return h_pool_fc
    print("Unknown exception, activation is ", activation)
```
因此可以简单地替换上面tf.layer中定义的层级函数。

替换后训练，成功率依然是20%左右，开始排查问题。



问题排查
========
（2021.7.12）对比别人实现的AlexNet，确认问题到底出在哪里。 
参考的目标是：
https://github.com/gholomia/AlexNet-Tensorflow/blob/master/src/alexnet.py
首先，逐层对比网络结构，看看是不是我的实现有误。


卷积层1和池化层1， 除了卷积层1的卷积核大小之外，实现均一致。
卷积层2和池化层2， 他设置的是[5,5,48,256]，而我设置成了[5,5,3,256]，其中48是96/2。这里是我手滑打错了……而且恰好3也可以被96整除，因此tensorflow没有报错。但修改后似乎也并没有什么显著提升。
卷积层3、4、5：他设置的是：
```
    "c3_filter": [3, 3, 256, 384],
    "c4_filter": [3, 3, 192, 384],
    "c5_filter": [3, 3, 192, 256]
```
192这里，我设置的都是384，改成他这个数字192测验一下，结果也没有什么变化。

全连接层有一个显著的区别是，他没有给任何层加ReLU，因此尝试去除我加到神经网络里的ReLU，去除后，准确率提升速度确实有肉眼可见的提升，但仍然很低，最终停留在35%左右。

因为之前没找到Tensorflow实现的SGD叫什么名字，所以一直在用AdamOptimizer，查阅手册发现sgd是MomentumOptimizer，将Adam换为MomentumOptimizer，即SGD+动量的方式，学习率0.1, 0.01, 0.001, 0.0001，动量=0.9, 0.45, 0.1均进行训练但效果似乎更差。换回AdamOptimizer。似乎差距应该只剩一个原因，就是图像增强。

在这一阶段中，我没有对原始图片做任何增强，而keras因为提供了比较简便的方法去做图片增强，所以它的训练是在图像增强之后做的。这应该是导致训练结果和keras差距较大的原因。
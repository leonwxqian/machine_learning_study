（2021.7.12）这次使用VGG16。训练集依然选用CIFAR-100。VGG16是2014年ILSVRC竞赛上VGGNet提出的模型中的一种，VGG-D和VGG-E因为各自有16层和19层，所以也分别称作VGG16和VGG19。VGG19比VGG16准确率更高，但相应的计算量也更大。本次我只试验VGG16。

VGG16的简化模型为：
卷积-卷积-池化  卷积-卷积-池化  卷积-卷积-卷积-池化  卷积-卷积-卷积-池化   卷积-卷积-卷积-池化   全连接-全连接-全连接
      2                2                3                    3                     3                       3
不算池化层，则一共16层。

相比之前的，VGG16多了很多batch normalization层。批标准化一般用在激活函数之前，使结果x=Wx+b各个维度均值为0，方差为1。通过标准化让激活函数分布在线性区间，让每一层的输入有一个稳定的分布，这会有利于网络的训练。
```
优点：
- 加大探索步长，加快收敛速度。
- 更容易跳出局部极小。
- 破坏原来的数据分布，一定程度上防止过拟合。
- 解决收敛速度慢和梯度爆炸。
————————————————
https://blog.csdn.net/eclipsesy/article/details/77597965
```

用于在每个训练步骤中强制每一层的输入具有大致相同的分布，在keras中是BatchNormalization()，而在tensorflow中是tf.layers.batch_normalization或是tf.nn.batch_normalization。

同时，VGG16大量使用l2正则化，根据网上介绍，l2会降低训练准确度，但会提高泛化程度：
```
In general, adding L2 regularization will have the effect of lowering your training accuracy and raising your generalization (test set) accuracy. 
https://stackoverflow.com/questions/47239837/tf-nn-l2-normalize-makes-accurracy-lower
```

按照作者的论文，可以实现下列VGG16的模型：

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 64)        1792      
activation (Activation)      (None, 32, 32, 64)        0         
batch_normalization (BatchNo (None, 32, 32, 64)        256       
dropout (Dropout)            (None, 32, 32, 64)        0 
conv2d_1 (Conv2D)            (None, 32, 32, 64)        36928     
activation_1 (Activation)    (None, 32, 32, 64)        0         
batch_normalization_1 (Batch (None, 32, 32, 64)        256       
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0    
conv2d_2 (Conv2D)            (None, 16, 16, 128)       73856     
activation_2 (Activation)    (None, 16, 16, 128)       0         
batch_normalization_2 (Batch (None, 16, 16, 128)       512       
dropout_1 (Dropout)          (None, 16, 16, 128)       0         
conv2d_3 (Conv2D)            (None, 16, 16, 128)       147584    
activation_3 (Activation)    (None, 16, 16, 128)       0         
batch_normalization_3 (Batch (None, 16, 16, 128)       512       
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0         
conv2d_4 (Conv2D)            (None, 8, 8, 256)         295168    
activation_4 (Activation)    (None, 8, 8, 256)         0         
batch_normalization_4 (Batch (None, 8, 8, 256)         1024      
dropout_2 (Dropout)          (None, 8, 8, 256)         0         
conv2d_5 (Conv2D)            (None, 8, 8, 256)         590080    
activation_5 (Activation)    (None, 8, 8, 256)         0         
batch_normalization_5 (Batch (None, 8, 8, 256)         1024      
dropout_3 (Dropout)          (None, 8, 8, 256)         0         
conv2d_6 (Conv2D)            (None, 8, 8, 256)         590080    
activation_6 (Activation)    (None, 8, 8, 256)         0         
batch_normalization_6 (Batch (None, 8, 8, 256)         1024      
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 256)         0         
conv2d_7 (Conv2D)            (None, 4, 4, 512)         1180160   
activation_7 (Activation)    (None, 4, 4, 512)         0         
batch_normalization_7 (Batch (None, 4, 4, 512)         2048      
dropout_4 (Dropout)          (None, 4, 4, 512)         0         
conv2d_8 (Conv2D)            (None, 4, 4, 512)         2359808   
activation_8 (Activation)    (None, 4, 4, 512)         0         
batch_normalization_8 (Batch (None, 4, 4, 512)         2048      
dropout_5 (Dropout)          (None, 4, 4, 512)         0         
conv2d_9 (Conv2D)            (None, 4, 4, 512)         2359808   
activation_9 (Activation)    (None, 4, 4, 512)         0         
batch_normalization_9 (Batch (None, 4, 4, 512)         2048      
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 512)         0         
conv2d_10 (Conv2D)           (None, 2, 2, 512)         2359808   
activation_10 (Activation)   (None, 2, 2, 512)         0         
batch_normalization_10 (Batc (None, 2, 2, 512)         2048      
dropout_6 (Dropout)          (None, 2, 2, 512)         0         
conv2d_11 (Conv2D)           (None, 2, 2, 512)         2359808   
activation_11 (Activation)   (None, 2, 2, 512)         0         
batch_normalization_11 (Batc (None, 2, 2, 512)         2048      
dropout_7 (Dropout)          (None, 2, 2, 512)         0         
conv2d_12 (Conv2D)           (None, 2, 2, 512)         2359808   
activation_12 (Activation)   (None, 2, 2, 512)         0         
batch_normalization_12 (Batc (None, 2, 2, 512)         2048      
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 512)         0         
dropout_8 (Dropout)          (None, 1, 1, 512)         0         
flatten (Flatten)            (None, 512)               0         
dense (Dense)                (None, 512)               262656    
activation_13 (Activation)   (None, 512)               0         
batch_normalization_13 (Batc (None, 512)               2048      
dropout_9 (Dropout)          (None, 512)               0         
dense_1 (Dense)              (None, 100)               51300     
activation_14 (Activation)   (None, 100)               0         
=================================================================
Total params: 15,047,588
Trainable params: 15,038,116
Non-trainable params: 9,472
_________________________________________________________________
Train on 50000 samples, validate on 10000 samples




使用tf.keras实现
==================
在我的笔记本RTX1060上确实性能不够，训练一个epoch需要75s左右，完整训练下来花费了2小时左右。为了后续修改时能比较快地看到结果，我这里贴了训练的75轮的训练精确度变化，这样后续我就可以大致根据前比如10～20轮的结果看到修改是否有效。因为到75轮时明显看出来网络在过拟合，所以就没有训练完预定的100轮了。

可以看到在CIFAR-100上取得了比AlexNet要好的结果，不过虽然训练集上准确度在50轮左右开始仍大幅上涨，但是在测试集上准确率却停滞甚至下滑，这证明网络出现了过拟合现象。因此，接下来使用tf.layer实现网络时，我将也着重处理这个过拟合的问题。
```
50000/50000 [==============================] - 81s 2ms/sample - loss: 7.5736 - acc: 0.0158 - val_loss: 6.6877 - val_acc: 0.0348
Epoch 2/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 6.9844 - acc: 0.0311 - val_loss: 6.5898 - val_acc: 0.0254
Epoch 3/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 6.6557 - acc: 0.0487 - val_loss: 6.4528 - val_acc: 0.0450
Epoch 4/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 6.4377 - acc: 0.0610 - val_loss: 6.4435 - val_acc: 0.0350
Epoch 5/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 6.2675 - acc: 0.0729 - val_loss: 6.4742 - val_acc: 0.0453
Epoch 6/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 6.1388 - acc: 0.0849 - val_loss: 6.6049 - val_acc: 0.0565
Epoch 7/100
50000/50000 [==============================] - 76s 2ms/sample - loss: 6.0013 - acc: 0.0934 - val_loss: 6.4452 - val_acc: 0.0792
Epoch 8/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 5.8829 - acc: 0.1053 - val_loss: 6.5115 - val_acc: 0.0697
Epoch 9/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 5.7726 - acc: 0.1146 - val_loss: 8.3475 - val_acc: 0.0882
Epoch 10/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 5.6551 - acc: 0.1266 - val_loss: 5.9755 - val_acc: 0.1190
Epoch 11/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 5.5652 - acc: 0.1370 - val_loss: 6.5043 - val_acc: 0.1267
Epoch 12/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 5.4324 - acc: 0.1513 - val_loss: 5.9959 - val_acc: 0.1291
Epoch 13/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 5.3224 - acc: 0.1659 - val_loss: 7.0162 - val_acc: 0.1314
Epoch 14/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 5.1891 - acc: 0.1799 - val_loss: 7.5895 - val_acc: 0.1383
Epoch 15/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 5.0706 - acc: 0.1966 - val_loss: 5.8836 - val_acc: 0.1892
Epoch 16/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 4.9545 - acc: 0.2086 - val_loss: 6.3265 - val_acc: 0.1887
Epoch 17/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 4.8289 - acc: 0.2304 - val_loss: 5.5596 - val_acc: 0.2280
Epoch 18/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 4.7115 - acc: 0.2437 - val_loss: 5.5338 - val_acc: 0.2373
Epoch 19/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 4.6061 - acc: 0.2578 - val_loss: 5.7941 - val_acc: 0.2337
Epoch 20/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 4.4908 - acc: 0.2768 - val_loss: 7.1203 - val_acc: 0.1827
Epoch 21/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 4.3877 - acc: 0.2934 - val_loss: 6.2950 - val_acc: 0.2685
Epoch 22/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 4.2960 - acc: 0.3083 - val_loss: 5.9313 - val_acc: 0.2815
Epoch 23/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 4.1949 - acc: 0.3238 - val_loss: 4.8981 - val_acc: 0.2898
Epoch 24/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 4.1028 - acc: 0.3404 - val_loss: 4.7327 - val_acc: 0.3170
Epoch 25/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 4.0132 - acc: 0.3549 - val_loss: 4.3837 - val_acc: 0.3488
Epoch 26/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.9285 - acc: 0.3708 - val_loss: 4.2674 - val_acc: 0.3683
Epoch 27/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 3.8545 - acc: 0.3894 - val_loss: 4.3957 - val_acc: 0.3552
Epoch 28/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 3.7798 - acc: 0.3989 - val_loss: 4.5649 - val_acc: 0.3422
Epoch 29/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 3.6976 - acc: 0.4124 - val_loss: 4.0254 - val_acc: 0.4053
Epoch 30/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 3.6325 - acc: 0.4249 - val_loss: 3.8966 - val_acc: 0.4186
Epoch 31/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.5667 - acc: 0.4355 - val_loss: 3.9948 - val_acc: 0.4044
Epoch 32/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.4976 - acc: 0.4489 - val_loss: 3.8511 - val_acc: 0.4264
Epoch 33/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.4359 - acc: 0.4545 - val_loss: 3.9119 - val_acc: 0.4202
Epoch 34/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.3774 - acc: 0.4648 - val_loss: 3.5829 - val_acc: 0.4624
Epoch 35/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.3277 - acc: 0.4764 - val_loss: 3.5430 - val_acc: 0.4675
Epoch 36/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.2723 - acc: 0.4875 - val_loss: 3.4746 - val_acc: 0.4852
Epoch 37/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.2109 - acc: 0.4967 - val_loss: 3.6497 - val_acc: 0.4598
Epoch 38/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.1499 - acc: 0.5104 - val_loss: 3.6317 - val_acc: 0.4703
Epoch 39/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.1120 - acc: 0.5139 - val_loss: 3.5288 - val_acc: 0.4739
Epoch 40/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.0560 - acc: 0.5233 - val_loss: 3.4585 - val_acc: 0.4848
Epoch 41/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 3.0041 - acc: 0.5337 - val_loss: 3.3755 - val_acc: 0.4983
Epoch 42/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.9653 - acc: 0.5396 - val_loss: 3.3556 - val_acc: 0.4930
Epoch 43/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.9141 - acc: 0.5481 - val_loss: 3.2075 - val_acc: 0.5193
Epoch 44/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.8689 - acc: 0.5531 - val_loss: 3.3217 - val_acc: 0.5129
Epoch 45/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.8289 - acc: 0.5634 - val_loss: 3.3066 - val_acc: 0.5038
Epoch 46/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.7866 - acc: 0.5730 - val_loss: 3.1782 - val_acc: 0.5234
Epoch 47/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.7341 - acc: 0.5812 - val_loss: 3.1886 - val_acc: 0.5209
Epoch 48/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.6908 - acc: 0.5886 - val_loss: 3.1632 - val_acc: 0.5292
Epoch 49/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.6622 - acc: 0.5927 - val_loss: 3.3727 - val_acc: 0.4988
Epoch 50/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 2.6107 - acc: 0.6024 - val_loss: 3.0938 - val_acc: 0.5397
Epoch 51/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.5760 - acc: 0.6090 - val_loss: 3.2364 - val_acc: 0.5182
Epoch 52/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 2.5400 - acc: 0.6149 - val_loss: 3.1718 - val_acc: 0.5325
Epoch 53/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 2.5133 - acc: 0.6201 - val_loss: 3.0436 - val_acc: 0.5472
Epoch 54/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 2.4786 - acc: 0.6268 - val_loss: 2.9850 - val_acc: 0.5594
Epoch 55/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 2.4362 - acc: 0.6377 - val_loss: 3.1014 - val_acc: 0.5397
Epoch 56/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 2.3983 - acc: 0.6421 - val_loss: 2.9784 - val_acc: 0.5617
Epoch 57/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 2.3763 - acc: 0.6469 - val_loss: 3.0204 - val_acc: 0.5581
Epoch 58/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 2.3344 - acc: 0.6542 - val_loss: 3.4906 - val_acc: 0.4965
Epoch 59/100
50000/50000 [==============================] - 76s 2ms/sample - loss: 2.3139 - acc: 0.6533 - val_loss: 3.0590 - val_acc: 0.5581
Epoch 60/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 2.2864 - acc: 0.6602 - val_loss: 2.9307 - val_acc: 0.5759
Epoch 61/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.2509 - acc: 0.6667 - val_loss: 2.9418 - val_acc: 0.5727
Epoch 62/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.2203 - acc: 0.6741 - val_loss: 2.9669 - val_acc: 0.5691
Epoch 63/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 2.1899 - acc: 0.6772 - val_loss: 3.0516 - val_acc: 0.5563
Epoch 64/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 2.1665 - acc: 0.6834 - val_loss: 2.9369 - val_acc: 0.5769
Epoch 65/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 2.1411 - acc: 0.6896 - val_loss: 2.8897 - val_acc: 0.5802
Epoch 66/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 2.1161 - acc: 0.6921 - val_loss: 2.9021 - val_acc: 0.5799
Epoch 67/100
50000/50000 [==============================] - 76s 2ms/sample - loss: 2.0811 - acc: 0.6987 - val_loss: 2.8792 - val_acc: 0.5814
Epoch 68/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 2.0562 - acc: 0.7068 - val_loss: 3.1189 - val_acc: 0.5487
Epoch 69/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 2.0375 - acc: 0.7087 - val_loss: 2.9824 - val_acc: 0.5730
Epoch 70/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 2.0095 - acc: 0.7154 - val_loss: 3.1076 - val_acc: 0.5524
Epoch 71/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 1.9866 - acc: 0.7176 - val_loss: 2.9712 - val_acc: 0.5697
Epoch 72/100
50000/50000 [==============================] - 76s 2ms/sample - loss: 1.9661 - acc: 0.7216 - val_loss: 2.9050 - val_acc: 0.5841
Epoch 73/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 1.9463 - acc: 0.7268 - val_loss: 2.8250 - val_acc: 0.5946
Epoch 74/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 1.9252 - acc: 0.7308 - val_loss: 2.8653 - val_acc: 0.5845
Epoch 75/100
50000/50000 [==============================] - 75s 2ms/sample - loss: 1.8909 - acc: 0.7386 - val_loss: 2.9753 - val_acc: 0.5690
```

使用tf.layer实现
==================
（2021.7.14）今天将tf.keras的实现转为tf.layer实现。 添加批量归一化层的对应实现：

```
def batch_normalization(inputs):
    return tf.compat.v1.layers.batch_normalization(
        inputs, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(), beta_regularizer=None,
        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
        training=False, trainable=True, name=None, reuse=None, renorm=False,
        renorm_clipping=None, renorm_momentum=0.99, fused=None, virtual_batch_size=None,
        adjustment=None
    )

```

同时，为了匹配全连接层和卷积层的l2正则化，则在层中添加对应代码。

```

def dense(input, units, stddev=0.04, kernel_regularizer='none', activation='relu'):
    if kernel_regularizer == 'none':
        return tf.layers.dense(inputs=input, units=units, use_bias=True,
                           kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                           bias_initializer=tf.constant_initializer(0.1), activation=tf.nn.relu)
    elif kernel_regularizer == 'l2' and activation == 'relu':
        return tf.layers.dense(inputs=input, units=units, use_bias=True,
                           kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                           bias_initializer=tf.constant_initializer(0.1), activation=tf.nn.relu,
                           kernel_regularizer=regularizers.l2(0.0005))
    elif kernel_regularizer == 'none' and activation == 'softmax':
        return tf.layers.dense(inputs=input, units=units, use_bias=True,
                           kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                           bias_initializer=tf.constant_initializer(0.1), activation=tf.nn.softmax )
```

在第一次转移之后出现了以下报错：
```
logits and labels must be broadcastable: logits_size=[11664,100] labels_size=[16,100]
```
这代表分类是正确的，而极有可能是整个神经网络的总体形状的值不对。打印了前几层的形状，与keras的形状对比，发现明显不一样，后查阅手册发现MaxPooling层keras默认会将strides设置为和kernel_size一样的值，但我实现的时候却默认设置成了1，因此导致了错误。

代码修改成下面的样子则可以正确地开始运行。
```
def max_pooling(input, kernel_size, strides=0):
    if strides == 0:
        strides = kernel_size
    return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1],
                          padding="VALID")
```



跑了几轮的成功率如下。显卡内存不够用了，只用了64个样本作测试，所以数字看起来很有规律，这也意味着似乎我的实现比keras的实现更复杂，因此仍有待分析。
```
轮次1： 0.015625
轮次2： 0.03125
轮次3： 0.046875
轮次4： 0.0625
```

与之前的训练结果类似，因此认为转移成功。
```
50000/50000 [==============================] - 81s 2ms/sample - loss: 7.5736 - acc: 0.0158 
Epoch 2/100
50000/50000 [==============================] - 75s 1ms/sample - loss: 6.9844 - acc: 0.0311 
Epoch 3/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 6.6557 - acc: 0.0487
Epoch 4/100
50000/50000 [==============================] - 74s 1ms/sample - loss: 6.4377 - acc: 0.0610
```


使用tf.nn实现
==================
（2021.7.14）因为之前已经大致提取出很多种与tf.layers等价的tf.nn的实现，包括卷积层、最大池化层、全连接层。因此这次只需要用tf.nn实现剩余几个新增的层，例如：批量归一化层，以及层级中用到的l2正则化等功能。

step 781, acc: 0.05
step 1562, acc: 0.04
step 2343, acc: 0.03
step 3124, acc: 0.04


但遇到和上面一样的问题即准确率只有10%左右，这和keras的相去甚远。 

调换l2和relu的顺序，
tf.nn.l2_normalize(tf.nn.relu(logits), dim=0, epsilon=0.0005)
设置
train_step = tf.compat.v1.train.MomentumOptimizer(0.01, 0.9).minimize(cross_entropy) 后好很多，但上涨速度仍不如keras的快。在制作VGG19时，我会先将这里面用到的层给抽出来，然后从keras转为layer或nn的实现便会变得很简单，因此也可以对比keras实现，确定问题是出在哪里。

```
学习率0.01， 动量0.9
epoch 1, acc: 0.022
epoch 2, acc: 0.021
epoch 3, acc: 0.019
epoch 4, acc: 0.021
epoch 5, acc: 0.02
epoch 6, acc: 0.038
epoch 7, acc: 0.032
epoch 8, acc: 0.034
epoch 9, acc: 0.029
epoch 10, acc: 0.036
epoch 11, acc: 0.031
epoch 12, acc: 0.046
epoch 13, acc: 0.045
epoch 14, acc: 0.043
epoch 15, acc: 0.044

学习率0.03，动量0.9 （每781个step为一个epoch）
step 781, acc: 0.021
step 1562, acc: 0.022
step 2343, acc: 0.03
step 3124, acc: 0.036
step 3905, acc: 0.044
step 4686, acc: 0.042
step 5467, acc: 0.027
step 6248, acc: 0.041

试验0.05， 0.07，0.09均不如0.03好，但仍慢于keras版本。


```
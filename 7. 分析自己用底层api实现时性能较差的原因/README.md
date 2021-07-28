这次只用最简单的CIFAR10+CNN的例子来确定问题到底在哪里。即使是对CIFAR10，我自己写的也不及官方的准确率高。

这一次我将分析keras的实现，做一个和它在网络层面上完全相同的实现。之后的差异可能就只是喂数据这里了，这个一步步来，后面会再仔细分析。

首先，将CNN的改为调用我的库函数，结果在第三轮即发生acc卡住不动的情况：

```
step 0, acc: 0.102
step 100, acc: 0.08
step 200, acc: 0.08
step 300, acc: 0.08
step 400, acc: 0.08
```

对比之前的代码，可以发现两者几乎无区别，唯一不一样的有两处，一是stddev的设置，二是各层bias的初始值设置，我先同步二者的stddev，但问题一样。

同步二者的bias，尤其是我之前dense层默认设置了bias，在输出层也有设置，这次将输出层的bias去掉，效果很明显，得到：
```
step 0, acc: 0.114
step 100, acc: 0.232
step 200, acc: 0.268
step 300, acc: 0.286
step 400, acc: 0.33
step 500, acc: 0.344
```

显然这是我的疏忽，最终的输出层就是要拟合或者分类的类别，是不会有bias的。加了bias后反而会干扰输出的准确性。

```
step 3500, acc: 0.546
step 3600, acc: 0.544
step 3700, acc: 0.558
step 3800, acc: 0.542
step 3900, acc: 0.544
step 4000, acc: 0.544
step 4100, acc: 0.546
step 4200, acc: 0.554
```

但这只解决了一个问题即acc不增长的问题，在42轮后，这种实现明显增长变慢，开始横向抖动，这是什么原因呢？

首先从卷积层入手，我先将wx+b的形式换成tf.layer，得到下列准确度变化：
```
step 1300, acc: 0.532
step 1400, acc: 0.5
step 1500, acc: 0.558
step 1600, acc: 0.532
step 1700, acc: 0.55
step 1800, acc: 0.526
……
step 2200, acc: 0.552
step 2300, acc: 0.548
step 2400, acc: 0.522
step 2500, acc: 0.536
step 2600, acc: 0.552
step 2700, acc: 0.544
step 2800, acc: 0.542
```

看起来“收敛”得更快了，在EPOCH 13左右就开始反复抖动。CIFAR10绝对不可能只有55%的准确率，接下来观察别人的代码，看下明显差异的地方，

发现代码在调用run()时应该和tfsess绑定，而不是和train_step绑定。
```
with tf.Session() as tfsess:
    tf.global_variables_initializer().run()
    for i in range(EPOCHES):
        (xb, yb) = get_batch(x_train, y_train, 500, i, 50000)
        tfsess.run([train_step, cross_entropy], feed_dict={x: xb, y_: yb})
```

修正后80 epoch准确率升到65%左右。 但是训练过程中明显发现随着轮次增加，训练速度越来越慢，查询资料可知每一轮速度变慢的原因是因为我在循环里修改了tf的运行图，因此将修改代码移出，问题解决。
```
https://stackoverflow.com/questions/47528401/tensorflow-why-my-code-is-running-slower-and-slower

https://blog.csdn.net/The_lastest/article/details/81130500
```

继续检查时也发现logits处编写的问题，官方文档`https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits`提到：

```
Warning: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency. Do not call this op with the output of softmax, as it will produce incorrect results.
```
因此将其移除，100轮之后准确率升至75%，这在不经任何特殊处理时认为是一个比较正常的准确率，和网上大家训练出来的类似。
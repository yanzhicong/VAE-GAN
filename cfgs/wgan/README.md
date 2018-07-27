# WGAN实验 cifar10数据生成 #


## 实验1： 

参数设置：
1. 生成器：1层全连接加3层反卷积，每层反卷积stride均为2，全连接输出(4,4,512)
2. 判别器：3层卷积加一层全连接，
3. 优化器：Adam优化器，learning rate=0.0001，beta1=0.5， beta2=0.9
   
实验结果：

![inception score](./res/cifar10exp1_inception_score.png)
![generator loss](./res/cifar10exp1_generator_loss.png)
![discriminator loss](./res/cifar10exp1_discriminator_loss.png)

问题记录：
1. 一直出现inception_score上不去的问题，论文提供代码的inception score能达到5.5左右，本实验中达到只能达到3.0到3.5之间
2. 在大约20k左右的时候，inception score开始稳定不变，貌似生成器生成图片的真实度已经无法再提升，随后生成器的损失逐渐增加，在23k的时候，生成器损失开始出现大幅震荡，判别器损失同时上升，inception score开始有下降的趋势。随后一直到100k，生成器损失一直在大幅震荡之中


问题修复:
1. 
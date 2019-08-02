# 使用cnn+rnn识别位置不定的验证码

## 依赖库

tensorflow-gpu-1.12.0

numpy

>当前需要在有GPU的机器上才能跑通，有心情再写个CPU版的

## 数据下载地址

https://pan.baidu.com/s/1mjI2Gxq

密码:d3iq

总共6w张

## 识别效果

验证码示例：

![yzm](https://github.com/shifop/yzm/blob/master/data/yzm/0a0y.jpg)

训练数据：54000条

测试数据：6000条

训练集准确率：93.1%

测试集准确率：87.7%

继续训练的话，测试集准确率应该能到90%以上

## 使用方法

训练：

```python
from model.simple_cnn import *
config = {
        "input_size": [30, 100, 3],  # 输入图片大小
        "tag_size": 4,  # 单个图片内含有的验证码个数
        "tag_list": "qwertyuiopasdfghjklzxcvbnm1234567890",  # 验证码包含的字符
        "split_rate": 10,  # 训练集和验证集分割比例
        "lr": 1e-5,  # 学习率
        "cnn_size": 50,  # 卷积网络大小
        "pool_size": [1, 9],  # 池化层参数
        "kernel_size": 5,  # 卷积宽度
        "rnn_size": 250,  # 卷积核数量
        "data_path": "../data",  # 训练数据存放目录，未处理过的数据放在该目录下的yzm目录
        "batch_size": 256,  # 每批训练数据数量
        "epochs": 2000,  # 训练次数
        "process": False,  # 是否将数据处理为record格式，第一次训练需要设置为True
        "print_interval": 1000,  # 每多少步输出一次准确率和loss
        "dev_interval": 2000,  # 每多少步在验证集上测试准确率
        "use_tensorboard": True
    }
oj = cnn(config)
oj.train("../save_model/model.ckpt")
```

调用训练好的模型预测
```python
from model.simple_cnn import *
config = {
        "input_size": [30, 100, 3],  # 输入图片大小
        "tag_size": 4,  # 单个图片内含有的验证码个数
        "tag_list": "qwertyuiopasdfghjklzxcvbnm1234567890",  # 验证码包含的字符
        "split_rate": 10,  # 训练集和验证集分割比例
        "lr": 1e-5,  # 学习率
        "cnn_size": 50,  # 卷积网络大小
        "pool_size": [1, 9],  # 池化层参数
        "kernel_size": 5,  # 卷积宽度
        "rnn_size": 250,  # 卷积核数量
        "data_path": "../data",  # 训练数据存放目录，未处理过的数据放在该目录下的yzm目录
        "batch_size": 256,  # 每批训练数据数量
        "epochs": 2000,  # 训练次数
        "process": False,  # 是否将数据处理为record格式，第一次训练需要设置为True
        "print_interval": 1000,  # 每多少步输出一次准确率和loss
        "dev_interval": 2000,  # 每多少步在验证集上测试准确率
        "use_tensorboard": True
    }
oj = cnn(config)
oj.load_mode("../save_model/model.ckpt")

# 读取图片
image = get_image("../data/yzm/0dxt.jpg")
print("预测结果是：%s"%(" ".join(oj.p(image))))
```

> 注：tensorflow存在BUG，不能释放资源。建议在程序中开启新的进程执行预测操作，当不在使用时，通过结束进程释放资源
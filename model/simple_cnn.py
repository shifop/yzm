import tensorflow as tf
import time
from tqdm import tqdm
import random

import tensorflow as tf
import numpy as np
import os
from datetime import timedelta
from PIL import Image


def get_image(path):
    img = np.array(Image.open(path))
    return img


def get_tag(tag, tag_list):
    index = [x for x in tag_list]
    rt = [index.index(x) for x in tag]
    return rt


class cnn(object):

    def __init__(self, config):
        self.config = config

        filenames = os.listdir(os.path.join(self.config["data_path"], 'yzm'))
        random.shuffle(filenames)

        # 分割验证集
        data_count = len(filenames) // self.config["batch_size"] if len(filenames) % self.config["batch_size"] == 0 \
            else len(filenames) // self.config["batch_size"] + 1
        dev_count = data_count // self.config['split_rate'] * self.config["batch_size"]
        train_count = len(filenames) - dev_count

        self.config["dev_count"] = dev_count
        self.config["train_count"] = train_count

        # 将数处理为record
        if self.config["process"]:
            tf.logging.info("将数据处理成record格式，提高训练时，GPU占用率")
            dev_filenames = filenames[:dev_count]
            train_filenams = filenames[dev_count:]
            # 保存
            self.process_data(os.path.join(self.config["data_path"], "yzm"), dev_filenames,
                              os.path.join(self.config["data_path"],
                                           "process/dev.record"))
            self.process_data(os.path.join(self.config["data_path"], "yzm"), train_filenams,
                              os.path.join(self.config["data_path"],
                                           "process/train.record"))
        tf.logging.info("开始构建模型")
        self.initial()

    def load_mode(self,path):
        sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                           gpu_options=tf.GPUOptions(allow_growth=True)))
        self.saver.restore(sess, path)

        # 生成id2tag
        self.id2tag = {}
        for index,x in enumerate(self.config["tag_list"]):
            self.id2tag[index] = x

    def process_data(self, path, filenames, save_path):
        writer = tf.python_io.TFRecordWriter(save_path)
        for index, filename in enumerate(tqdm(filenames)):
            tag = get_tag(filename.split('.')[0], self.config['tag_list'])
            filename = os.path.join(path, filename)
            image = get_image(filename)
            features = tf.train.Features(feature={
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[np.array(image, np.float32).tostring()])),
                "tag": tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(tag, np.int64)))
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
        writer.close()

    def get_data(self, path, batch_size, is_train):
        def parser(record):
            features = tf.parse_single_example(record,
                                               features={
                                                   'image': tf.FixedLenFeature([], tf.string),
                                                   'tag': tf.FixedLenFeature([self.config["tag_size"]], tf.int64)
                                               }
                                               )
            return features['image'], features['tag']

        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        if is_train:
            dataset = dataset.shuffle(self.config["batch_size"] * 10)
            dataset = dataset.prefetch(self.config["batch_size"])
        iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        image, tag = iter.get_next()

        tag = tf.cast(tag, tf.int32)
        image = tf.decode_raw(image, tf.float32)

        image = tf.reshape(image, [-1] + self.config["input_size"])
        tag = tf.reshape(tag, [-1, self.config["tag_size"]])
        return image, tag, iter.make_initializer(dataset)

    def forecast(self, input, cell, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # 一层卷积，处理后数据shape变为,batch_szie, 1,(L-5)+1,cnn_size
            feature = tf.layers.conv2d(input, self.config["cnn_size"], \
                                       [self.config["input_size"][0], self.config["kernel_size"]],
                                       activation=tf.nn.relu)

            # 池化，处理后数据shape变为,batch_szie, 1,(L-7)+1,cnn_size
            feature = tf.layers.max_pooling2d(feature, self.config["pool_size"], [1, 1], name='max_pool')
            shape = [self.config["input_size"][1] - self.config["kernel_size"] -
                     self.config["pool_size"][1] + 2, self.config["cnn_size"]]

            feature = tf.reshape(feature, shape=[-1] + shape)
            feature = tf.split(feature, self.config['tag_size'], axis=1)
            feature = [tf.reshape(x, [-1, 1, (shape[0] // self.config['tag_size']) * shape[1]]) for x in feature]
            feature = tf.concat(feature, axis=1)
            ft_dropout = tf.layers.dropout(cell(feature), rate=0.5, training=False)
            ft = tf.layers.dense(ft_dropout, len(self.config["tag_list"]), name='dense', reuse=tf.AUTO_REUSE)

            p = tf.nn.softmax(ft, axis=-1)
            y_pred_cls = tf.argmax(p, -1)
            return y_pred_cls

    def create_model(self, input, tag, cell, is_trainning, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            label_one_hot = tf.one_hot(tag, len(self.config["tag_list"]))

            # 一层卷积，处理后数据shape变为,batch_szie, 1,(L-5)+1,cnn_size
            feature = tf.layers.conv2d(input, self.config["cnn_size"], \
                                       [self.config["input_size"][0], self.config["kernel_size"]],
                                       activation=tf.nn.relu)

            # 池化，处理后数据shape变为,batch_szie, 1,(L-7)+1,cnn_size
            feature = tf.layers.max_pooling2d(feature, self.config["pool_size"], [1, 1], name='max_pool')
            shape = [self.config["input_size"][1] - self.config["kernel_size"] -
                     self.config["pool_size"][1] + 2, self.config["cnn_size"]]

            feature = tf.reshape(feature, shape=[-1] + shape)
            feature = tf.split(feature, self.config['tag_size'], axis=1)
            feature = [tf.reshape(x, [-1, 1, (shape[0] // self.config['tag_size']) * shape[1]]) for x in feature]
            feature = tf.concat(feature, axis=1)
            ft_dropout = tf.layers.dropout(cell(feature), rate=0.5, training=is_trainning)
            ft = tf.layers.dense(ft_dropout, len(self.config["tag_list"]), name='dense', reuse=tf.AUTO_REUSE)

            p = tf.nn.softmax(ft, axis=-1)
            y_pred_cls = tf.argmax(p, -1)
            correct_pred = tf.equal(tf.argmax(label_one_hot, -1), y_pred_cls)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # 计算loss
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=ft, labels=label_one_hot)
            loss = tf.reduce_mean(loss)

            return acc, loss

    def initial(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32, [None] + self.config["input_size"], name='tag')
            self.tag = tf.placeholder(tf.int32, [None, self.config["tag_size"]], name='tag')
            self.is_trainning = tf.placeholder(tf.bool, (), name='is_trainning')

            self.train_input, self.train_tag, self.train_data_op = self.get_data(
                os.path.join(self.config["data_path"], "process/train.record"), self.config["batch_size"],
                is_train=True)

            self.dev_input, self.dev_tag, self.dev_data_op = self.get_data(
                os.path.join(self.config["data_path"], "process/dev.record"), self.config["batch_size"], is_train=True)

            cell = tf.keras.layers.Bidirectional(
                tf.keras.layers.CuDNNLSTM(self.config["rnn_size"], return_sequences=True))

            self.train_acc, self.train_loss,_ = self.create_model(self.train_input, self.train_tag, cell, True, "cnn_rnn")
            self.dev_acc, self.dev_loss,_ = self.create_model(self.dev_input, self.dev_tag, cell, True, "cnn_rnn")

            self.y_pred_cls = self.forecast(self.input, cell, "cnn_rnn")

            self.summary_train_loss = tf.summary.scalar('train_loss', self.train_loss)
            self.summary_train_acc = tf.summary.scalar('train_acc', self.train_acc)
            self.summary_dev_loss = tf.summary.scalar('dev_loss', self.dev_loss)
            self.summary_dev_acc = tf.summary.scalar('dev_acc', self.dev_acc)

            # 创建
            self.optimer = tf.train.AdamOptimizer(self.config["lr"]).minimize(self.train_loss)
            self.saver = tf.train.Saver()
            for index, x in enumerate(tf.trainable_variables()):
                print("%d:%s" % (index, x))

    def train(self, save_path):
        tf.logging.info("开始训练模型")
        start_time = time.time()
        total_batch = 0  # 总批次
        min_loss = -1
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 90000  # 如果超过指定轮未提升，提前结束训练
        all_loss = 0
        all_acc = 0

        log_writer = None
        if self.config["use_tensorboard"]:
            log_writer = tf.summary.FileWriter('../log')

        flag = False
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            sess.run(tf.global_variables_initializer())
            # self.saver.restore(sess, load_path)

            train_count = self.config["train_count"]
            dev_count = self.config["dev_count"]
            steps = train_count // self.config["batch_size"] if train_count % self.config["batch_size"] == 0 \
                else train_count // self.config["batch_size"] + 1

            for epoch in range(self.config['epochs']):
                # 初始化输入数据
                sess.run(self.train_data_op)
                if flag:
                    break
                for step in range(steps):
                    if total_batch % self.config["print_interval"] == 0:
                        if total_batch % self.config["dev_interval"] == 0 and total_batch != 0:
                            # 跑验证集
                            dev_loss, dev_acc = self.evaluate(sess, dev_count, log_writer,
                                                              total_batch // self.config["dev_interval"])
                            if min_loss == -1 or min_loss <= dev_acc:
                                self.saver.save(sess=sess, save_path=save_path)
                                improved_str = '*'
                                last_improved = total_batch
                                min_loss = dev_acc
                            else:
                                improved_str = ''

                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train acc: {2:>6.3}, Val Loss:{3:>6.2}, ' \
                                  'Val acc:{4:>6.3} Time: {5} {6}'
                            print(msg.format(total_batch, all_loss / self.config["print_interval"],
                                             all_acc / self.config["print_interval"], dev_loss,
                                             dev_acc, time_dif, improved_str))
                        else:
                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train acc: {2:>6.3}, Time: {3}'
                            print(msg.format(total_batch, all_loss / self.config["print_interval"],
                                             all_acc / self.config["print_interval"], time_dif))
                        all_loss = 0
                        all_acc = 0

                    if self.config["use_tensorboard"]:
                        train_loss, train_acc, summary_train_loss, summary_train_acc, _ = sess.run([self.train_loss,
                                                                                                    self.train_acc,
                                                                                                    self.summary_train_loss,
                                                                                                    self.summary_train_acc,
                                                                                                    self.optimer])
                        log_writer.add_summary(summary_train_loss, total_batch)
                        log_writer.add_summary(summary_train_acc, total_batch)
                    else:
                        train_loss, train_acc, _ = sess.run([self.train_loss, self.train_acc, self.optimer])
                    all_loss += train_loss
                    all_acc += train_acc
                    total_batch += 1

                    if total_batch - last_improved > require_improvement:
                        # 验证集正确率长期不提升，提前结束训练
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break  # 跳出循环
                if flag:
                    break

    def evaluate(self, sess, dev_count, log_writer, cur_step):
        sess.run(self.dev_data_op)
        rt_loss = []
        rt_acc = []
        steps = dev_count // self.config["batch_size"] if dev_count % self.config["batch_size"] == 0 \
            else dev_count // self.config["batch_size"] + 1
        for step in range(steps):
            if self.config["use_tensorboard"]:
                dev_loss, dev_acc, summary_dev_loss, summary_dev_acc = sess.run(
                    [self.dev_loss, self.dev_acc, self.summary_dev_loss, self.summary_dev_acc])

                log_writer.add_summary(summary_dev_loss, cur_step*steps + step)
                log_writer.add_summary(summary_dev_acc, cur_step*steps + step)

            else:
                dev_loss, dev_acc = sess.run([self.dev_loss, self.dev_acc])
            rt_loss.append(dev_loss)
            rt_acc.append(dev_acc)
        return sum(rt_loss) / len(rt_loss), sum(rt_acc) / len(rt_acc)

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def p(self,image):
        sess = self.sess
        id2tag = self.id2tag
        data = {self.input: image}
        rt = sess.run(self.y_pred_cls, feed_dict=data)
        return [id2tag[x] for x in rt]


def train():
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
        "use_tensorboard": True #是否将训练日志写入tensorboard
    }
    oj = cnn(config)
    oj.train("../save_model/")

def forecast():
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
        "use_tensorboard": True #是否将训练日志写入tensorboard
    }
    oj = cnn(config)
    oj.load_mode("../save_model/model.ckpt")

    # 读取图片
    image = get_image("../data/yzm/0dxt.jpg")
    print("预测结果是：%s"%(" ".join(oj.p(image))))

if __name__ == '__main__':
    # 训练模型
    # train()
    # 使用训练好的模型进行预测
    forecast()

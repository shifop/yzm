import tensorflow as tf
import time
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import os
from datetime import timedelta
from PIL import Image


def get_image(path):
    img = np.array(Image.open(path))
    return img

def get_tag(tag):
    index=[x for x in "qwertyuiopasdfghjklzxcvbnm1234567890"]
    rt = [index.index(x) for x in tag]
    return rt


class cnn(object):

    def __init__(self, config):
        self.config = config
        self.__create_model()

    def get_data(self,path):
        filenames =os.listdir(path)
        images = []
        tags = []
        for index, filename in enumerate(filenames):
            tag=get_tag(filename.split('.')[0])
            filename = os.path.join(path, filename)
            images.append(get_image(filename))
            tags.append(tag)
            if len(tag)!=4:
                print('%d:%s'%(index, ''.join(tag)))
        return images,tags

    def __create_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32, [None, 90, 250, 3], name='tag')
            self.tag = tf.placeholder(tf.int32, [None,4], name='tag')
            self.is_trainning = tf.placeholder(tf.bool, (), name='is_trainning')
            label_one_hot = tf.one_hot(self.tag, 42)
            feature = tf.layers.conv2d(self.input,10,[90,5], activation=tf.nn.relu)
            feature = tf.layers.max_pooling2d(feature, [1,7],[1,1], name='max_pool')
            feature = tf.reshape(feature,shape=[-1,240,10])
            feature = tf.split(feature,4, axis=1)
            feature = [tf.reshape(x,[-1,1,600]) for x in feature]
            feature = tf.concat(feature, axis=1)
            cell = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(50, return_sequences=True))
            ft_dropout = tf.layers.dropout(cell(feature), rate = 0.5, training=self.is_trainning)
            self.ft = tf.layers.dense(ft_dropout,42, name='dense', reuse=tf.AUTO_REUSE)

            p = tf.nn.softmax(self.ft, axis=-1)
            y_pred_cls = tf.argmax(p, -1)
            correct_pred = tf.equal(tf.argmax(label_one_hot, -1), y_pred_cls)
            self.acc = tf.reduce_mean( tf.cast(correct_pred, tf.float32))

            # 计算loss
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.ft, labels=label_one_hot)
            self.loss = tf.reduce_mean(self.loss)
            self.optimer = tf.train.AdamOptimizer(0.00001).minimize(self.loss)
            self.saver = tf.train.Saver()

    def train(self, save_path):
        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        min_loss = -1
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 90000  # 如果超过指定轮未提升，提前结束训练
        all_loss = 0

        flag = False
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            sess.run(tf.global_variables_initializer())
            # self.saver.restore(sess, load_path)

            images, tags = self.get_data('../data/yzm')

            dev_images = images[:20]
            dev_tags = tags[:20]

            dev_data = {self.input:dev_images,self.tag:dev_tags, self.is_trainning:False}

            images = images[20:]
            tags = tags[20:]

            size = len(images)

            for epoch in range(200000):
                if flag:
                    break
                for step in range(size//20):
                    input = images[step*20:(step+1)*20]
                    tag = tags[step*20:(step+1)*20]
                    data = {self.input:input,self.tag:tag,self.is_trainning:True}
                    if total_batch % 1000 == 0:
                        if total_batch % 2000 == 0 and total_batch!=0:
                            # 跑验证集
                            dev_acc = self.evaluate(sess, dev_data)
                            if min_loss == -1 or min_loss <= dev_acc:
                                self.saver.save(sess=sess, save_path=save_path)
                                improved_str = '*'
                                last_improved = total_batch
                                min_loss = dev_acc
                            else:
                                improved_str = ''

                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Val acc:{2:>6.3} Time: {3} {4}'
                            print(msg.format(total_batch, all_loss / 1000, dev_acc, time_dif, improved_str))
                        else:
                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Time: {2}'
                            print(msg.format(total_batch, all_loss / 1000, time_dif))
                        all_loss = 0

                    loss_train,_ = sess.run([self.acc, self.optimer],feed_dict=data)
                    all_loss += loss_train
                    total_batch += 1

                    if total_batch - last_improved > require_improvement:
                        # 验证集正确率长期不提升，提前结束训练
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break  # 跳出循环
                if flag:
                    break

    def evaluate(self,sess,data):
        acc = sess.run(self.acc, feed_dict=data)
        return acc

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

if __name__=='__main__':
    oj = cnn(None)
    oj.train("../save_model/")
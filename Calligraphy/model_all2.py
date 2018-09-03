#!/usr/bin/env python3
#_*_coding:utf-8_*_

import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import numpy as np
import threading, queue
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import utils


EP_MAX = 20000
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 5             # loop update operation n-steps
n_model = 1
S_DIM = 64 * 64
A_DIM = 3
A_BOUND = [50, 50, 50]
width = 64
height = 64
count = 0


class ppo_agent():
    def __init__(self, ckpt_dir='./ckpt', sum_dir='./summaries',
                 epsilon=0.2, A_LR=0.0001, C_LR=0.0005, D_LR=0.0002):
        self.ckpt_dir = ckpt_dir
        self.replay = []
        self.EPSILON = epsilon  # Clipped surrogate objective
        self.A_LR = A_LR  # learning rate for actor
        self.C_LR = C_LR  # learning rate for critic
        self.D_LR = D_LR  # learning rate for discriminator

        self.sess = tf.Session()
        self._build_ph()

        # critic
        self.v, self.num= self._build_value()
        self.advantage = self.ph_dr - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        self.ls, pi, pi_params = self._build_policy('pi', trainable=True)
        oldls, oldpi, oldpi_params = self._build_policy('oldpi', trainable=False)
        self.type = tf.argmax(pi[0], 1)
        self.sample_op = tf.concat([tf.expand_dims(self.type, axis=-1),
                                    tf.squeeze(tf.cast(pi[1].sample(1), tf.int64), axis=0)], axis=1)  # choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        ratio = tf.concat([pi[1].prob(self.ph_a[:, 1:4]) / (oldpi[1].prob(self.ph_a[:, 1:4]) + 1e-5),
                          pi[0] / oldpi[0]], axis=1)
        surr = ratio * self.ph_adv   # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(surr,
            tf.clip_by_value(ratio, 1. - self.EPSILON, 1. + self.EPSILON) * self.ph_adv))
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)


        # number and cmd
        self.nloss = tf.reduce_mean(tf.square((self.num - self.ph_num) / (self.ph_num + 1e5)))
        self.ntrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.nloss)

        # d
        self.r_logit = self._build_discriminator(self.real)
        self.f_logit = self._build_discriminator(self.ph_s)

        # losses
        def gradient_penalty(real, fake, f):
            def interpolate(a, b):
                shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
                alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.get_shape().as_list())
                return inter

            x = interpolate(real, fake)
            pred = f(x)
            gradients = tf.gradients(pred, x)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
            gp = tf.reduce_mean((slopes - 1.) ** 2)
            return gp

        wd = tf.reduce_mean(self.r_logit) - tf.reduce_mean(self.f_logit)
        gp = gradient_penalty(self.real, self.ph_s, self._build_discriminator)
        self.d_loss = -wd + gp * 10.0
        self.dtrain_op = tf.train.AdamOptimizer(D_LR).minimize(self.d_loss)

        self.load()

        self.saver = tf.train.Saver(max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(sum_dir, self.sess.graph)

    def _build_ph(self):
        self.ph_s = tf.placeholder(tf.float32, [None, 64, 64, 1], 'state')
        self.ph_dr = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.ph_la = tf.placeholder(tf.float32, [None, 4], 'last_a')
        self.ph_a = tf.placeholder(tf.float32, [None, 4], 'action')
        self.ph_adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.real = tf.placeholder(tf.float32, [None, 64, 64, 1], 'real')
        self.ph_num = tf.placeholder(tf.float32, [None, 1], 'Number')
        self.ph_ls = tf.placeholder(tf.float32, [None, 128], 'last_state')

    def _build_value(self):
        with tf.variable_scope('critic'):
            with tf.variable_scope('feature'):
                cnn1 = tf.layers.conv2d(self.ph_s, 32, [3, 3], strides=2,
                                        activation=tf.nn.leaky_relu, name='cnn1')
                cnn2 = tf.layers.conv2d(cnn1, 32, [3, 3], strides=2,
                                        activation=tf.nn.leaky_relu, name='cnn2')
                cnn3 = tf.layers.conv2d(cnn2, 32, [3, 3], strides=2,
                                        activation=tf.nn.leaky_relu, name='cnn3')
                fl1 = tf.layers.flatten(cnn3, name='fl1')

            with tf.variable_scope('value'):
                dn1 = tf.layers.dense(fl1, 1, name='dn1')

            with tf.variable_scope('num'):
                dn2 = tf.layers.dense(fl1, 1, activation=tf.nn.relu, name='dn2')
        return dn1, dn2

    def _build_policy(self, name, trainable):
        with tf.variable_scope(name):
            with tf.variable_scope('feature'):
                cnn1 = tf.layers.conv2d(self.ph_s, 32, [3, 3], strides=2, trainable=trainable,
                                        activation=tf.nn.leaky_relu, name='cnn1')
                cnn2 = tf.layers.conv2d(cnn1, 32, [3, 3], strides=2, trainable=trainable,
                                        activation=tf.nn.leaky_relu, name='cnn2')
                cnn3 = tf.layers.conv2d(cnn2, 32, [3, 3], strides=2, trainable=trainable,
                                        activation=tf.nn.leaky_relu, name='cnn3')
                fl1 = tf.layers.flatten(cnn3, name='fl1')

            with tf.variable_scope('cmd'):
                dn1 = tf.layers.dense(fl1, 128, activation=tf.nn.relu,
                                      trainable=trainable, name='dn1')
                dn1 += self.ph_ls
                dn2 = tf.layers.dense(dn1, 35, activation=tf.nn.softmax,
                                      trainable=trainable, name='dn2')
                mu = A_BOUND * tf.layers.dense(dn1, A_DIM, tf.nn.sigmoid, trainable=trainable, name='mu')
                sigma = tf.layers.dense(dn1, A_DIM, tf.nn.softplus, trainable=trainable, name='sigma')
                cmd = Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dn1, [dn2, cmd], params


    def _build_discriminator(self, img):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            cnn1 = tf.layers.conv2d(img, 32, [3, 3], strides=2,
                                    activation=tf.nn.leaky_relu, name='cnn1')
            cnn2 = tf.layers.conv2d(cnn1, 32, [3, 3], strides=2,
                                    activation=tf.nn.leaky_relu, name='cnn2')
            cnn3 = tf.layers.conv2d(cnn2, 32, [3, 3], strides=2,
                                    activation=tf.nn.leaky_relu, name='cnn3')
            pool1 = tf.layers.max_pooling2d(cnn3, 3, 1, name='pool1')
            fl1 = tf.layers.flatten(pool1, name='fl1')
            dn1 = tf.layers.dense(fl1, 32, activation=tf.nn.leaky_relu, name='dn1')
            dn2 = tf.layers.dense(dn1, 1, name='dn2')

        return dn2

    def connect(self, s_o, s):
        alpha = 0.618
        return ((1 - alpha) * s_o + s * alpha) / 2

    def act(self, obs, ls):
        feed_dict = {
            self.ph_s: [obs],
            self.ph_ls: [ls]
            # self.ph_la: [obs[1]],
        }
        cmd, ls_ = self.sess.run([self.sample_op, self.ls], feed_dict=feed_dict)
        return np.squeeze(cmd, axis=0), np.squeeze(ls_, axis=0)

    def number(self, obs):
        feed_dict = {
            self.ph_s: [obs],
            # self.ph_la: [obs[1]],
        }
        num = self.sess.run(self.num, feed_dict=feed_dict)
        num = np.squeeze(num, axis=0)
        return int(num + 1)

    def reward(self, obs):
        feed_dict = {
            self.ph_s: [obs],
            # self.ph_la: [obs[1]],
        }
        reward = self.sess.run(self.f_logit, feed_dict=feed_dict)
        return np.squeeze(reward, axis=0)

    def get_v(self, s):
        return self.sess.run(self.v, {self.ph_s: [s]})[0, 0]

    def dtrain(self, s_o, s):
        feed_dict = {
            self.ph_s: s,
            self.real: s_o,
        }
        _, dloss = self.sess.run([self.dtrain_op, self.d_loss], feed_dict=feed_dict)
        return np.min(dloss)

    def ntrain(self, s, num, cmd):
        feed_dict = {
            self.ph_num: [np.expand_dims(float(num), axis=-1)],
            self.ph_s: [s],
            self.ph_la: [cmd[1:4]]
        }
        _ = self.sess.run(self.ntrain_op, feed_dict=feed_dict)

    def Draw(self, state, cmd):
        state = np.squeeze(state, axis=-1)
        img = (state * 255 + 255) / 2
        im = Image.fromarray(np.uint8(img))
        cmd = list(cmd)
        cmd[1] = cmd[1].astype(np.int32)
        word = chr(cmd[0] + s_bh)
        size = cmd[1]
        loc_x = cmd[2]
        loc_y = cmd[3]
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype("NotoSerifCJKsc-SemiBold.otf", size)
        dr.text((loc_x, loc_y), word, font=font, fill='#000000')
        img_x = np.expand_dims((np.array(im) * 2 - 255) / 255, axis=-1)

        return img_x

    def DateSet(self, start, end):
        data_pool = []

        for codepoint in range(int(start), int(end)):
            word = chr(codepoint)
            im = Image.new('L', (width, height), 255)
            dr = ImageDraw.Draw(im)
            font = ImageFont.truetype("NotoSerifCJKsc-SemiBold.otf", 50)
            dr.text((1, 1), word, font=font, fill='#000000')
            data_pool.append((np.array(im) * 2 - 255) / 255)
        data_pool = np.expand_dims(np.array(data_pool), axis=-1)
        return data_pool

    def save(self, epoch_times):
        global count
        if count % epoch_times == 0:
            save_path = self.saver.save(self.sess, '%s/Epoch_(%d)_(%d).ckpt' %
                                   (self.ckpt_dir, count/epoch_times, epoch_times))
            print('Model saved in file: % s' % save_path)
        count += 1

    def load(self):
        ckpt_dir = self.ckpt_dir
        utils.mkdir(ckpt_dir + '/')
        if not utils.load_checkpoint(ckpt_dir, self.sess):
            self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.sess.run(self.update_oldpi_op)  # old pi to pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                data = np.vstack(data)
                s_o, s, a, r, ls = data[:, :S_DIM].reshape([-1, 64, 64, 1]), \
                                   data[:, S_DIM:2*S_DIM].reshape([-1, 64, 64, 1]), \
                                   data[:, 2*S_DIM: 2*S_DIM + 4],\
                                   data[:, 2*S_DIM+4: 2*S_DIM+5], data[:, 2*S_DIM+5:]
                _ = self.dtrain(s_o, s)
                adv = self.sess.run(self.advantage, {self.ph_s: s, self.ph_dr: r})
                [self.sess.run(self.atrain_op, {self.ph_s: s, self.ph_a: a, self.ph_adv: adv,
                                                self.ph_ls: ls}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.ph_s: s, self.ph_dr: r}) for _ in
                 range(UPDATE_STEP)]
                self.save(300)
                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, \
            DATASET_ALL, ls
        while not COORD.should_stop():
            s = Image.new('L', (width, height), 255)
            s = np.expand_dims((np.array(s)*2-255)/255, axis=-1)
            ep_r = 0
            buffer_s, buffer_a, buffer_r, buffer_ls, buffer_so = [], [], [], [],[]
            s_o = DATASET_ALL[GLOBAL_UPDATE_COUNTER]
            num = 64
            for t in range(num):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer
                a, ls_ = self.ppo.act(self.ppo.connect(s_o, s), ls)
                s_ = self.ppo.Draw(s, a)
                r = self.ppo.reward(s_)
                buffer_ls.append(ls)
                buffer_s.append(s.flatten())
                buffer_so.append(s_o.flatten())
                buffer_a.append(a)
                buffer_r.append(r)  # normalize reward, find to be useful
                s = s_
                ls = ls_
                ep_r += r
                GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size
                if t == num - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bso, bs, ba, br, bls = np.vstack(buffer_so),\
                                           np.vstack(buffer_s),\
                                           np.vstack(buffer_a),\
                                           np.array(discounted_r),\
                                           np.vstack(buffer_ls)
                    buffer_s, buffer_a, buffer_r, buffer_ls, buffer_so = [], [], [], [],[]
                    QUEUE.put(np.hstack((bso, bs, ba, br, bls)))
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break
            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)
       

if __name__ == '__main__':
    GLOBAL_PPO = ppo_agent()
    start, end = (0x4E00, 0x9FA5)  # 常见汉字
    DATASET_ALL = GLOBAL_PPO.DateSet(start, end)
    s_bh, e_bh = (0x31C0, 0x31E3)  # Unicode笔画
    DATASET_BH = GLOBAL_PPO.DateSet(s_bh, e_bh)
    ls = np.zeros(128)
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # no update now
    ROLLING_EVENT.set()  # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()
    Replay_d = []
    dloss = 0
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
    threads[-1].start()
    COORD.join(threads)

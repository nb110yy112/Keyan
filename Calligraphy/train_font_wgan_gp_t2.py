# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import utils
import traceback
import numpy as np
import tensorflow as tf
import models_font_t2 as models
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random

""" param """
epoch = 10000
batch_size = 32
lr = 0.0002
z_dim = 100
n_critic = 3
gpu_id = 3
count = 0

''' data '''
width = 60
height = 60
s_bh, e_bh = (0x31C0, 0x31E3)
start, end = (0x4E00, 0x9FA5)   #常见汉字
data_pool = []
replay = []

# for v in range(6, 51, 1):
for codepoint in range(int(start), int(end)):
    word = chr(codepoint)
    im = Image.new('L', (width, height), 255)
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype("NotoSerifCJKsc-SemiBold.otf", 50)
    dr.text((0, 0), word, font=font, fill='#000000')
    data_pool.append((np.array(im)*2-255)/255)
data_pool = np.expand_dims(np.array(data_pool), axis=-1)
lth = len(data_pool)
""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' models '''
    generator = models.generator
    discriminator = models.discriminator_wgan_gp

    ''' graph '''
    # inputs
    real = tf.placeholder(tf.float32, shape=[None, 60, 60, 1])
    z = tf.placeholder(tf.float32, shape=[None, 60, 60, 1])
    num_label = tf.placeholder(tf.float32, shape=[None, 1])
    cmd_label = tf.placeholder(tf.float32, shape=[None, 100, 4])

    # generate
    fake, num, cmd = generator(z, reuse=False)

    # dicriminate
    r_logit = discriminator(real, reuse=False)
    f_logit = discriminator(fake)

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
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
    gp = gradient_penalty(real, fake, discriminator)
    d_loss = -wd + gp * 10.0
    g_loss = -tf.reduce_mean(f_logit)
    num_loss = tf.nn.l2_loss(num - num_label)
    cmd_loss = tf.reduce_mean(tf.nn.l2_loss(cmd - cmd_label))

    # otpims
    d_var = utils.trainable_variables('discriminator')
    g_var = utils.trainable_variables('generator')
    num_var = utils.trainable_variables('generator/num')
    cmd_var = utils.trainable_variables('generator/cmd')
    d_step = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_var)
    g_step = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_var)
    num_step = tf.train.AdamOptimizer().minimize(num_loss, var_list=num_var)
    cmd_step = tf.train.AdamOptimizer().minimize(cmd_loss, var_list=cmd_var)

    # summaries
    d_summary = utils.summary({wd: 'wd', gp: 'gp', d_loss: 'd_loss'})
    g_summary = utils.summary({g_loss: 'g_loss'})
    num_summary = utils.summary({num_loss: 'num_loss'})
    cmd_summary = utils.summary({cmd_loss: 'cmd_loss'})

    # sample
    f_sample, _, _ = generator(z, training=False)


""" train """
''' init '''
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
summary_writer = tf.summary.FileWriter('./summaries/mnist_wgan_gp', sess.graph)

''' initialization '''
ckpt_dir = './checkpoints/mnist_wgan_gp'
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

''' train '''
try:
    z_ipt_sample = data_pool[0:64]

    # batch_epoch = lth // (batch_size * n_critic)
    batch_epoch = 10
    max_it = epoch * batch_epoch

    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)
        if len(replay) > 100:
            replay.pop(0)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        be = count % lth
        if (be + batch_size) > lth:
            real_ipt = np.concatenate((data_pool[be:lth], data_pool[0:(be + batch_size - lth)]), axis=0)
        else:
            real_ipt = data_pool[be:(be + batch_size)]
        num_opt, cmd_opt = sess.run([num, cmd], feed_dict={z: real_ipt})
        img_x = []
        for k in range(batch_size):
            im = Image.new('L', (width, height), 255)
            for j in range(int(num_opt[k] * 100 + 1)):
                word = chr(int(cmd_opt[k][j][0] * 35) + s_bh)
                size = int(cmd_opt[k][j][1] * 25) + 26
                loc_x = int(cmd_opt[k][j][2] * 30)
                loc_y = int(cmd_opt[k][j][3] * 30)
                dr = ImageDraw.Draw(im)
                font = ImageFont.truetype("NotoSerifCJKsc-SemiBold.otf", size)
                dr.text((loc_x, loc_y), word, font=font, fill='#000000')
            img_x.append((np.array(im) * 2 - 255) / 255)
        img_x = np.expand_dims(np.array(img_x), axis=-1)
        replay.append((img_x, real_ipt, num_opt, cmd_opt))

        # train D
        for i in range(n_critic):
            rand = np.random.choice(len(replay))
            img_x, real_ipt, num_opt, cmd_opt = replay[rand]
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: img_x, z: real_ipt})
            count = count + 32
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        rand = np.random.choice(len(replay))
        img_x, real_ipt, num_opt, cmd_opt = replay[rand]
        z_ipt = real_ipt
        if count % 32 == 0:
            num_summary_opt, _, cmd_summary_opt, _ = sess.run([num_summary, num_step, cmd_summary, cmd_step], feed_dict={
                z: img_x, num_label: num_opt, cmd_label: cmd_opt})
            summary_writer.add_summary(num_summary_opt, it)
            summary_writer.add_summary(cmd_summary_opt, it)

        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: img_x})
        summary_writer.add_summary(g_summary_opt, it)

        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
        summary_writer.add_summary(g_summary_opt, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            count = count % lth
            f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample})
            save_dir = './sample_images_while_training/mnist_wgan_gp'
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(f_sample_opt, 8, 8), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))

except Exception as e:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()

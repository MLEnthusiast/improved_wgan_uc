import extract_data as dd

import os, sys
sys.path.append(os.getcwd())
import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.plot


DIM = 64
LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 64  # Batch size
ITERS = 200000  # How many generator iterations to train for
OUTPUT_DIM = 12288  # Number of pixels (3*64*64)

lib.print_model_settings(locals().copy())


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)


def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*DIM, noise)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 8*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*DIM, 4*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs):

    #output = tf.reshape(inputs, [-1, 64, 64, 3])
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
    output = LeakyReLU(output)


    output = tf.reshape(output, [-1, 4*4*8*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*DIM, 1, output)

    return tf.reshape(output, [-1])


real_data_in = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
real_data = 2*(real_data_in-.5)
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

# Standard WGAN loss
gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

# Gradient penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1],
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# For generating samples
fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples_128 = Generator(128, noise=fixed_noise_128)
def generate_image(frame):
    samples = session.run(fixed_noise_samples_128)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((128, 3, 64, 64)), './out/samples_{}.jpg'.format(frame))


# Dataset iterators
def random_train_batch(dataset):
    idx = np.random.choice(_num_images_train, size=BATCH_SIZE, replace=False)
    train_img = dataset[idx, :, :, :]
    train_img = np.reshape(train_img, newshape=[BATCH_SIZE, OUTPUT_DIM])
    return train_img

# Train loop
with tf.Session() as session:
    start = 0
    if not os.path.exists('./out/'):
        os.makedirs('./out/')

    # Directory where data is present
    data_path = 'uc_data/'

    # filename
    filename = 'uc_data.tar.gz'

    dd._extract(data_path, filename)

    images_train = np.load(os.path.join(data_path, 'images_train_batch_1.npy'))
    _num_images_train = len(images_train)

    session.run(tf.global_variables_initializer())

    for iteration in xrange(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)
        # Train critic
        disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            train_imgs = random_train_batch(images_train)
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_in: train_imgs})

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            generate_image(iteration)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
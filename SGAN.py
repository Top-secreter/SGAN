import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers
from astropy.io import fits

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"
# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)
# The dimension of our random noise vector.
random_dim = 900

def load_data(): # load the data     
    x_train = pd.read_csv('data_O.csv',header=None,dtype=np.float32)
    x_train = np.array(x_train)
    return (x_train)   
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5) 
def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    return x
def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(3700, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return generator
def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=3700, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return discriminator

def get_gan_network(discriminator, random_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the 
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 900-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the spectrum is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def save_generated_images(epoch, generator, examples= 200, figsize=(1, 3700)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)    
    np.savetxt('D:\\fake_DA_%d.csv'% epoch, generated_images,delimiter=',')
    generated_images = generated_images.reshape(examples, 1, 3700)

def similarity_O ( data ):
    similarity = []
    f = data.shape[0]
    for j in range(f):
      b = []
      for i in range(3):
        file_path = "D:\\similarity\\O\\%s"%i + ".fits"
        O = fits.open(file_path)
        O_data = np.flipud(O[0].data)
        O_data = O_data[4][0:3700]
        O_data = MaxMinNormalization(O_data)
        data[j] = MaxMinNormalization(data[j])
        c = data[j]-O_data
        d = O_data
        a = c/d
        where_are_nan = np.isnan(a)
        where_are_inf = np.isinf(a)
        a[where_are_nan] = 0
        a[where_are_inf] = 0
        similarity_n = np.std(a, ddof = 1)
        b.append(similarity_n)
      similarity.append(min(b))  
    similarity_mean = np.mean(similarity)
    return similarity_mean
    
def train(epochs=1, batch_size=2):
    # Get the training and testing data
    x_train = load_data()
    # Split the training data into batches of minisize 
    batch_count = int(x_train.shape[0] / batch_size)
    # Build our SGAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)
    b=[]
    c=[]
    d=[]

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and spectra
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            # Generate fake spectra
            generated_images = generator.predict(noise)
            #print (list(set(spectra_batch)-set(generated_spectra)))
            X = np.concatenate([image_batch, generated_images])
            # Labels for generated and real data
            y_dis = np.random.uniform(0, 0.3, size=[6, 1])
            # One-sided label smoothing
            y_dis[:batch_size] = np.random.uniform(0.9, 1.2, size=[3, 1])
            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)
            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)
            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

        noise = np.random.normal(0, 1, size=[156, random_dim])
        generated_images = generator.predict(noise)    
        loss = similarity_O(generated_images)
        real_pre = np.mean(discriminator.predict(x_train))
        fake_pre = np.mean(discriminator.predict(generated_images))
        b.append(loss)
        c.append(real_pre)
        d.append(fake_pre)
        loss_save = np.array(b)
        real_pre_save = np.array(c)
        fake_pre_save = np.array(d)
#        if e > 0 and e % 10 == 0:        
#          if not os.path.exists("keras_model_O"):
#            os.makedirs("keras_model_O")
#          generator.save_weights("keras_model_O/G_model%d.hdf5" % e,True)
#          discriminator.save_weights("keras_model_O/D_model%d.hdf5" % e,True)
        if  e % 500 == 0:
          save_generated_images(e, generator)
                
    np.savetxt('D:\\GAN\\d_loss_O.csv', loss_save,delimiter=',')
    np.savetxt('D:\\GAN\\real_pre_O.csv', real_pre_save,delimiter=',')
    np.savetxt('D:\\GAN\\fake_pre_O.csv', fake_pre_save,delimiter=',')

#if __name__ == '__main__':
train(8000, 3)
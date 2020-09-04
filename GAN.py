

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from absl.flags import FLAGS


from tensorflow.keras.datasets import mnist


flags.DEFINE_string('path', '', 'path to the directory where the generated image is to be stored')



(X_train, y_train), (X_test, y_test) = mnist.load_data()



# ## Filtering out the Data for Faster Training on Smaller Dataset



only_zeros = X_train[y_train==0]




only_zeros.shape





import tensorflow as tf
from tensorflow.keras.layers import Dense,Reshape,Flatten
from tensorflow.keras.models import Sequential





np.random.seed(42)
tf.random.set_seed(42)

codings_size = 100



generator = Sequential()
generator.add(Dense(100, activation="relu", input_shape=[codings_size]))
generator.add(Dense(150,activation='relu'))
generator.add(Dense(784, activation="sigmoid")) # 28*28 = 784
generator.add(Reshape([28,28]))





discriminator = Sequential()
discriminator.add(Flatten(input_shape=[28,28]))
discriminator.add(Dense(150,activation='relu'))
discriminator.add(Dense(100,activation='relu'))
discriminator.add(Dense(1,activation="sigmoid"))

discriminator.compile(loss="binary_crossentropy", optimizer="adam")



GAN = Sequential([generator, discriminator])





discriminator.trainable = False




GAN.compile(loss="binary_crossentropy", optimizer="adam")




# ### Setting up Training Batches




import tensorflow as tf





batch_size = 32

# The buffer_size in Dataset.shuffle() can affect the randomness of your dataset, and hence the order in which elements are produced. 




# my_data = X_train
my_data = only_zeros





dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)




dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)




epochs = 1


# **NOTE: The generator never actually sees any real images. It learns by viewing the gradients going back through the discriminator. The better the discrimnator gets through training, the more information the discriminator contains in its gradients, which means the generator can being to make progress in learning how to generate fake images, in our case, fake zeros.**
# 
# Training Loop



# Grab the seprate components
generator, discriminator = GAN.layers

# For every epcoh
for epoch in range(epochs):
    print(f"Currently on Epoch {epoch+1}")
    i = 0
    # For every batch in the dataset
    for X_batch in dataset:
        i=i+1
        if i%100 == 0:
            print(f"\tCurrently on batch number {i} of {len(my_data)//batch_size}")
        #####################################
        ## TRAINING THE DISCRIMINATOR ######
        ###################################
        
        # Create Noise
        noise = tf.random.normal(shape=[batch_size, codings_size])
        
        # Generate numbers based just on noise input
        gen_images = generator(noise)
        
        # Concatenate Generated Images against the Real Ones
        # TO use tf.concat, the data types must match!
        X_fake_vs_real = tf.concat([gen_images, tf.dtypes.cast(X_batch,tf.float32)], axis=0)
        
        # Targets set to zero for fake images and 1 for real images
        y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
        
        # This gets rid of a Keras warning
        discriminator.trainable = True
        
        # Train the discriminator on this batch
        discriminator.train_on_batch(X_fake_vs_real, y1)
        
        
        #####################################
        ## TRAINING THE GENERATOR     ######
        ###################################
        
        # Create some noise
        noise = tf.random.normal(shape=[batch_size, codings_size])
        
        # We want discriminator to belive that fake images are real
        y2 = tf.constant([[1.]] * batch_size)
        
        # Avois a warning
        discriminator.trainable = False
        
        GAN.train_on_batch(noise, y2)
        
print("TRAINING COMPLETE")            




# Most likely your generator will only learn to create one type of noisey zero
# Regardless of what noise is passed in.




noise = tf.random.normal(shape=[10, codings_size])




image = generator(noise)

plt.imsave(FLAGS.path,image)


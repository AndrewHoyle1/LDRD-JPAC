# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 01:07:20 2021

@author: Andrew Hoyle 
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import Constraint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import click
import datetime
import os
import io
import time
import mceg

def dot(A,B):
    return A[0]*B[0] - A[1]*B[1] - A[2]*B[2] - A[3]*B[3]

def add(A,B):
    ApB={}
    ApB[0]=A[0]+B[0] 
    ApB[1]=A[1]+B[1] 
    ApB[2]=A[2]+B[2] 
    ApB[3]=A[3]+B[3] 
    return ApB

def sub(A,B):
    AmB={}
    AmB[0]=A[0]-B[0] 
    AmB[1]=A[1]-B[1] 
    AmB[2]=A[2]-B[2] 
    AmB[3]=A[3]-B[3] 
    return AmB

def generate_samples(batch):
    buffer = 1000
    nsamples=10000
    samples=mceg.gen_samples(nsamples)
    
    pip={}  
    pip[0]=samples[:,12]
    pip[1]=samples[:,3]
    pip[2]=samples[:,6]
    pip[3]=samples[:,9]

    # get pi- 4-momentum
    pim={}  
    pim[0]=samples[:,13]
    pim[1]=samples[:,4]
    pim[2]=samples[:,7]
    pim[3]=samples[:,10]

    # add pim and pip 4-vectors (eventwise)
    pipPpim=add(pim,pip)

    # compute the pi+pi- invariant mass (eventwise)
    Mpipi=np.sqrt(dot(pipPpim,pipPpim))
    
    H,E=np.histogram(Mpipi,range=(0.5,1.0),bins=50)
    
    m_list = []
    rho_list = []
    drho_list = []

    for i in range(nsamples):
        m=0.5*(E[1:]+E[:-1])
        rho =H/np.sum(nsamples)
        m_list.append(m)
        rho_list.append(rho)
        drho=np.sqrt(H)/np.sum(nsamples)
        drho_list.append(drho)
    
    m_train, m_test, rho_train, rho_test, drho_train, drho_test = train_test_split(m_list, rho_list, drho_list, shuffle = True,
                                            test_size = 0.25)
    
    m_train, m_val, rho_train, rho_val, drho_train, drho_val = train_test_split(m_train, rho_train, drho_train, shuffle = True,
                                            test_size = 0.25)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((m_train, rho_train, drho_train)).shuffle(buffer).batch(batch)
    val_dataset = tf.data.Dataset.from_tensor_slices((m_val, rho_val, drho_val)).shuffle(buffer).batch(batch)
    test_dataset = tf.data.Dataset.from_tensor_slices((m_test, rho_test, drho_test)).shuffle(buffer).batch(batch)
    
    return train_dataset, val_dataset, test_dataset

def wasserstein_loss(y_true, y_pred):
    return tf.math.reduce_mean(y_true * y_pred)
    
def dist(inp):
    m = inp[0]
    params = inp[1]
    A = params[:,0:1]
    B = params[:,1:2]
    C = params[:,2:3]
    G = params[:,3:4]
    s = params[:,4:5]
    
    return A / ((m**2 - s)**2 + (m*G)**2 ) + B*s + C
    
def generator():
    init = tf.random_normal_initializer(0,0.02)
    
    n_nodes = 512
    
    noise = layers.Input((1000), dtype = tf.float64)
    m = layers.Input((50), dtype = tf.float64)
    
    Dense1 = layers.Dense(n_nodes, kernel_initializer = init)(noise)
    activation1 = layers.LeakyReLU()(Dense1)
    Dense2 = layers.Dense(n_nodes, kernel_initializer = init)(activation1)
    activation2 = layers.LeakyReLU()(Dense2)
    Dense3 = layers.Dense(n_nodes, kernel_initializer = init)(activation2)
    activation3 = layers.LeakyReLU()(Dense3)
    
    param_outputs = layers.Dense(5)(activation3)
    
    #param_output = layers.concatenate([A,B,C,G,s])
    
    dist_output = layers.Lambda(dist)([m, param_outputs])
    
    param_model = tf.keras.Model(inputs = noise, outputs = param_outputs)
    dist_model = tf.keras.Model(inputs = [noise, m], outputs = dist_output)
    
    return param_model, dist_model

Gen_Param, Gen_Dist = generator()

def discriminator():
     init = tf.random_normal_initializer(0, 0.02)
    
     inp = layers.Input((50), dtype = tf.float32)
     n_nodes = 256
    
     Dense1 = layers.Dense(n_nodes, kernel_initializer = init)(inp)
     activation1 = layers.LeakyReLU()(Dense1)
     Dense2 = layers.Dense(n_nodes, kernel_initializer = init)(activation1)
     activation2 = layers.LeakyReLU()(Dense2)
    
     output = layers.Dense(1)(activation2)
    
     model = tf.keras.Model(inputs = inp, outputs = output)
    
     return model
 
Discriminator = discriminator()

mse_loss = tf.keras.losses.MeanSquaredError()

def generator_loss(gen_surface):
    y = tf.ones_like(gen_surface)
    #return wasserstein_loss(gen_surface, y)
    return mse_loss(gen_surface, y)

def discriminator_loss(gen_surface, real_surface):
    x = -1*tf.ones_like(gen_surface)
    #disc_loss_fake = wasserstein_loss(gen_surface, x)
    disc_loss_fake = mse_loss(gen_surface, x)
    
    y = tf.ones_like(real_surface)
    #disc_loss_real = wasserstein_loss(real_surface,y)
    disc_loss_real = mse_loss(real_surface, y)
    
    total_loss = (disc_loss_real + disc_loss_fake)
    return total_loss

def gradient_penalty(batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        real_images = tf.cast(real_images, tf.float32)
        
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = Discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

@tf.function
def train_step(m, rho, n_disc, epoch, g_opt, d_opt, batch, gp_weight):
    for i in range(n_disc):
        noise = tf.random.normal([len(m),1000])
        
        with tf.GradientTape() as tape:
            gen_rho = Gen_Dist([noise,m] , training = True)
            
            fake_disc = Discriminator(gen_rho, training = True)
            real_disc = Discriminator(rho, training = True)
            
            d_loss = discriminator_loss(fake_disc, real_disc)
            
            gp = gradient_penalty(len(m), rho, gen_rho)
            
            d_loss += gp * gp_weight
            
        d_gradient = tape.gradient(d_loss, Discriminator.trainable_variables)
        
        d_opt.apply_gradients(zip(d_gradient, Discriminator.trainable_variables))
    
    noise = tf.random.normal([len(m),1000])
    with tf.GradientTape() as tape:
        gen_rho = Gen_Dist([noise,m] , training = True)
        
        fake_disc = Discriminator(gen_rho, training = True)
        
        g_loss = generator_loss(fake_disc)
    
    gen_gradient = tape.gradient(g_loss, Gen_Dist.trainable_variables)
    
    g_opt.apply_gradients(zip(gen_gradient, Gen_Dist.trainable_variables))
    
    with summary_writer.as_default():
        tf.summary.scalar("Gen Loss", g_loss, step = epoch)
        tf.summary.scalar("Disc Loss", d_loss, step = epoch)

    return g_loss, d_loss

def cast_to_img(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    
    plt.close(figure)
    buf.seek(0)
    
    image = tf.image.decode_png(buf.getvalue(), channels = 4)
    
    image = tf.expand_dims(image,0)
    
    return image

def distribution_graphs(m, rho, drho, epoch):
    figure = plt.figure(figsize = (20, 10))
    nrows, ncols = 1,2
    
    noise = tf.random.normal([len(m), 1000])
    rho_gen = Gen_Dist([noise,m], training = False)
    #print(rho_gen.shape)
    rho_gen_mean = np.mean(rho_gen, axis = 0)
    rho_gen_std = np.std(rho_gen, axis = 0)
    down = rho_gen_mean-rho_gen_std
    up = rho_gen_mean + rho_gen_std
    
    ax=plt.subplot(nrows,ncols,1)
    #ax.errorbar(m,rho,drho,fmt='k.')
    ax.errorbar(m[0], rho[0], drho[0], fmt= 'k.', zorder = 10)
    ax.fill_between(m[0], down, up, color = 'y')
    ax.tick_params(axis='both', which='both',labelsize=20,direction='in',length=10)
    ax.set_xlabel(r'$M_{\pi\pi}$',size=20);
    ax.set_ylabel(r'$\rho(M_{\pi\pi})$',size=20);
    
    ax = plt.subplot(nrows, ncols, 2)
    ax.errorbar(m[0], rho[0]/rho_gen_mean, drho[0]/rho_gen_mean, fmt = 'k.', zorder = 10)
    ax.fill_between(m[0], down/rho_gen_mean, up/rho_gen_mean, color = 'y')
    ax.tick_params(axis='both', which='both',labelsize=20,direction='in',length=10)
    ax.set_xlabel(r'$M_{\pi\pi}$',size=20);
    ax.set_ylabel(r'$\rho(M_{\pi\pi})$/$\rho_gen(M_{\pi\pi})/$',size=20);
    
    with summary_writer.as_default():
        tf.summary.image("Prediction Images", cast_to_img(figure), step = epoch)
        
def parameter_graphs(m, rho, epoch):
    noise = tf.random.normal([len(m),1000])
    replicas = Gen_Param(noise, training = False)
    
    nrows,ncols=5,5
    fig = plt.figure(figsize=(ncols*7,nrows*5))
    cnt=0
    for i in range(5):
        for j in range(5):
            if j>i:
                cnt+=5-j
                break
            cnt+=1
            #print(i,j,cnt)
            ax=plt.subplot(nrows,ncols,cnt)
            if i==j: color='r'
            else: color='k'
            ax.plot(replicas[:,j],replicas[:,i],ls='',marker = 'o',color=color)
            ax.set_xlabel(r'$\rm param~{%d}$'%j,size=20);
            ax.set_ylabel(r'$\rm param~{%d}$'%i,size=20);
            ax.tick_params(axis='both', which='both',labelsize=20,direction='in',length=10)
            
    with summary_writer.as_default():
        tf.summary.image("Parameter Correlations", cast_to_img(fig), step = epoch)
            
log_dir = "Dist_Logs/"
summary_writer = tf.summary.create_file_writer(log_dir + 'fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
@click.command()
@click.option('--num_epochs', type=int, default = 10000)
@click.option('--num_discriminator', type = int, default = 1)
@click.option('--patience', type=int, default = 50)
@click.option('--gen_lr', type = float, default = 5e-6)
@click.option('--disc_lr', type = float, default = 5e-6)
@click.option('--batch_size', type=int, default = 1028)
@click.option('--gp_weight', type = int, default = 10)
def train(num_epochs, patience,
          gen_lr, disc_lr,
          num_discriminator, batch_size, gp_weight):
    
    print("producing distributions")
    
    train_dataset, val_dataset, test_dataset = generate_samples(batch_size)
    
    print("data unpacked")
    
    best_epoch = 0
    best_loss = 10000
    
    g_opt = tf.keras.optimizers.RMSprop(lr = gen_lr)
    d_opt = tf.keras.optimizers.RMSprop(lr = disc_lr)
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(genrator_optimizer = g_opt,
                                 discriminator_optimizer = d_opt,
                                 gen = Gen_Dist,
                                 disc = Discriminator)
    
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for i in range(num_epochs):
        start = time.time()
        
        gen_loss_list = []
        gen_val_loss_list = []
        disc_loss_list = []
        disc_val_loss_list = []
        
        print("creating distribution graphic")
        
        if i%10 == 0:
            for (m,rho,drho) in val_dataset.take(1):
                distribution_graphs(m, rho, drho, i)
                parameter_graphs(m,rho,i)
                checkpoint.save(file_prefix = checkpoint_prefix + str(i+1))
        
        print("evaluating epoch {}".format(i+1))
        print("training")
        for n, (m, rho, drho) in train_dataset.enumerate():            
            #tf.summary.trace_on(graph=True, profiler = True)
            
            gen_loss, disc_loss = train_step(m, rho, num_discriminator, i,
                                             g_opt, d_opt, batch_size, gp_weight)
            
            gen_loss = gen_loss.numpy()
            gen_loss_list.append(gen_loss)
            disc_loss = disc_loss.numpy()
            disc_loss_list.append(disc_loss)
        
        print("validating")
        for n, (m, rho, drho) in val_dataset.enumerate():
            gen_loss, disc_loss = train_step(m, rho, num_discriminator, i,
                                             g_opt, d_opt, batch_size, gp_weight)
            
            gen_loss = gen_loss.numpy()
            gen_val_loss_list.append(gen_loss)
            disc_loss = disc_loss.numpy()
            disc_val_loss_list.append(disc_loss)
            
        avg_gen_loss = np.mean(gen_loss_list)
        avg_disc_loss = np.mean(disc_loss_list)
        avg_gen_val_loss = np.mean(gen_val_loss_list)
        avg_disc_val_loss = np.mean(disc_val_loss_list)
        
        if (np.abs(avg_disc_loss) < best_loss):
            best_epoch = i + 1
            best_loss = np.abs(avg_disc_loss)
            checkpoint.save(file_prefix = checkpoint_prefix + str(i+1))
        """elif (i + 1 >= best_epoch + patience):
            print("There has been no significant improvement since epoch ", best_epoch, ".")
            break"""

        print ('Time for epoch {} is {} sec'.format(i + 1, time.time()-start))
        print("Best epoch so far: " , best_epoch)
        print("Avg Generator Loss: ", avg_gen_loss)
        print("Avg Discriminator Loss: ", avg_disc_loss)
        print("Avg Validation Generator Loss: ", avg_gen_val_loss)
        print("Avg Validation Discriminator Loss: ", avg_disc_val_loss)


train()
#print(Generator.summary())
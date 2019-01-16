from keras.models import Sequential,  Model
from keras.layers import Convolution2D,Input,BatchNormalization,Conv2D,Activation,Lambda,Subtract,Conv2DTranspose, PReLU
from keras.regularizers import l2
from keras.layers import  Reshape,Dense,Flatten
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt

def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def SRCNN_model():

    input_shape = (72,14,1)
    x = Input(shape = input_shape)
    c1 = Convolution2D( 64 , 9 , 9 , activation = 'relu', init = 'he_normal', border_mode='same')(x)
    c2 = Convolution2D( 32 , 1 , 1 , activation = 'relu', init = 'he_normal', border_mode='same')(c1)
    c3 = Convolution2D( 1 , 5 , 5 , init = 'he_normal', border_mode='same')(c2)
    #c4 = Input(shape = input_shape)(c3)
    model = Model(input = x, output = c3)
    ##compile
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error']) 
    return model
  

def SRCNN_predict_model():
  
    input_shape = (72,14,1)
    x = Input(shape = input_shape)
    c1 = Convolution2D( 64 , 9 , 9 , activation = 'relu', init = 'he_normal', border_mode='same')(x)
    c2 = Convolution2D( 32 , 1 , 1 , activation = 'relu', init = 'he_normal', border_mode='same')(c1)
    c3 = Convolution2D( 1 , 5 , 5 , init = 'he_normal', border_mode='same')(c2)
    #c4 = Input(shape = input_shape)(c3)
    model = Model(input = x, output = c3)
    ##compile
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error']) 
    return model
  
def SRCNN_train(train_data ,train_label, val_data , val_label):
    srcnn_model = SRCNN_model()
    print(srcnn_model.summary())
    
    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(train_data, train_label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs= 300 , verbose=0)
    
    #srcnn_model.save_weights("drive/codes/my_srcnn/SRCNN_SUI5_weights/SRCNN_48_12.h5")
    srcnn_model.save_weights("drive/codes/my_srcnn/SRCNN_VehA_weights/SRCNN_36_22.h5")
    # srcnn_model.load_weights("m_model_adam.h5")


def SRCNN_predict(input_data):
    srcnn_model = SRCNN_predict_model()
    srcnn_model.load_weights("drive/codes/my_srcnn/SRCNN_VehA_weights/SRCNN_36_22.h5")
    
    #srcnn_model.load_weights("drive/codes/my_srcnn/SRCNN_SUI5_weights/SRCNN_48_12.h5")
    predicted  = srcnn_model.predict(input_data)
    return predicted

  
def DNCNN_model ():
  
    inpt = Input(shape=(None,None,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(18):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error']) 
    
    
    return model

def DNCNN_train(train_data ,train_label, val_data , val_label):
  
  dncnn_model = DNCNN_model()
  print(dncnn_model.summary())

  checkpoint = ModelCheckpoint("DNCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                               save_weights_only=False, mode='min')
  callbacks_list = [checkpoint]

  dncnn_model.fit(train_data, train_label, batch_size=128, validation_data=(val_data, val_label),
                  callbacks=callbacks_list, shuffle=True, epochs= 200 , verbose=0)
  dncnn_model.save_weights("drive/codes/my_srcnn/DNCNN_VehA_weights/DNCNN_36_22.h5")
  #dncnn_model.save_weights("drive/codes/my_srcnn/DNCNN_SUI5_weights/DNCNN_48_12.h5")
  # srcnn_model.load_weights("m_model_adam.h5")
  
  
  
def DNCNN_predict(input_data):
  dncnn_model = DNCNN_model()
  dncnn_model.load_weights("drive/codes/my_srcnn/DNCNN_weights/DNCNN_24_12.h5")
  predicted  = dncnn_model.predict(input_data)
  return predicted



if __name__ == "__main__":
    # load datasets 
    
    interp_noisy = numpy.load('drive/codes/my_srcnn/results_22_36_rbf.npy')
    perfect = loadmat('drive/codes/my_srcnn/Perfect_H_40000.mat')['My_perfect_H']
    
    #interp_noisy = numpy.load('drive/codes/my_srcnn/SUI5_12_48_rbf.npy')
    #perfect = loadmat('drive/codes/my_srcnn/SUI5_perfect.mat')['SUI5_perfect_H']
    
    
    perfect_image = numpy.zeros((40000,72,14,2))
    perfect_image[:,:,:,0] = numpy.real(perfect)
    perfect_image[:,:,:,1] = numpy.imag(perfect)
    perfect_image = numpy.concatenate((perfect_image[:,:,:,0], perfect_image[:,:,:,1]), axis=0).reshape(80000, 72, 14, 1)
    
    
    ####### ------ training SRCNN ------ #######
    idx_random = numpy.random.rand(80000) < 0.9
    train_data, train_label = interp_noisy[idx_random,:,:,:] , perfect_image[idx_random,:,:,:]
    val_data, val_label = interp_noisy[~idx_random,:,:,:] , perfect_image[~idx_random,:,:,:]    
    SRCNN_train(train_data ,train_label, val_data , val_label)
    
   
    ####### ------ prediction using SRCNN ------ #######
    srcnn_predicted = SRCNN_predict(interp_noisy)
    ## saving the output of the trained model to train the denoising network 
    numpy.save('drive/codes/my_srcnn/SRCNN_outputs/srcnn_out_36_22.npy',srcnn_predicted)
    
    #numpy.save('drive/codes/my_srcnn/SRCNN_outputs/srcnn_SUI_48_12.npy',srcnn_predicted)
    
    '''
    srcnn_model = SRCNN_predict_model()
    srcnn_model.load_weights('drive/codes/my_srcnn/SRCNN_weights/SRCNN_24_22.h5')
    srcnn_predicted = srcnn_model.predict(interp_noisy)
    numpy.save('drive/codes/my_srcnn/SRCNN_outputs/srcnn_out_16_22.npy',srcnn_predicted)
    
    dncnn_model = DNCNN_model()
    dncnn_model.load_weights('drive/codes/my_srcnn/DNCNN_weights/DNCNN_16_22.h5')
    dncnn_predicted = dncnn_model.predict(srcnn_predicted)
    
    #srcnn_out = numpy.load('drive/codes/my_srcnn/SRCNN_outputs/srcnn_out_16_22.npy')
    
    '''
    ####### ------ traing DNCNN using SRCNN outputs ------ ######
    
    
    
    idx_random = numpy.random.rand(80000) < 0.9
    ## load the SRCNN output 
    srcnn_out = numpy.load('drive/codes/my_srcnn/SRCNN_outputs/srcnn_out_36_22.npy')
    
    #srcnn_out = numpy.load('drive/codes/my_srcnn/SRCNN_outputs/srcnn_SUI_48_12.npy')
    train_data, train_label = srcnn_out[idx_random,:,:,:] , perfect_image[idx_random,:,:,:]
    val_data, val_label = srcnn_out[~idx_random,:,:,:] , perfect_image[~idx_random,:,:,:]
    
    DNCNN_train(train_data ,train_label, val_data , val_label)
    
    
    
   
    
    
    ## plot the results 
    
    '''
    n = 4    
    fig = plt.figure(figsize=(10,10))
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3*i+1)
        srcnn_pred =  interp_noisy[i+40,:,:,0].squeeze()
        ax.imshow(srcnn_pred)
        
        ax = fig.add_subplot(n, 3, 3*i+2)
        dncnn_pred =  srcnn_predicted[i+40,:,:,0].squeeze()
        ax.imshow(dncnn_pred)
        
        ax = fig.add_subplot(n, 3, 3*i+3)
        X_label = perfect_image[i+40,:,:,0].squeeze()
        ax.imshow(X_label)
        
    '''
   

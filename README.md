# ChannelNet
Implementation of the paper "Deep Learning-Based Channel Estimation" https://arxiv.org/abs/1810.05893

# Abstract 

In this paper, we present a deep learning (DL) algorithm for channel estimation in communication systems. We consider the time-frequency response of a fast fading communication channel as a two-dimensional image. The aim is to find the unknown values of the channel response  using some known values at the pilot locations. To this end, a general pipeline using deep image processing techniques, image super-resolution (SR) and image restoration (IR) is proposed. This scheme considers the pilot values, altogether, as a low-resolution image and uses an SR network cascaded with a denoising IR network to estimate the channel. Moreover, an implementation of the proposed pipeline is presented. The estimation error shows that the presented algorithm is comparable to the minimum mean square error (MMSE) with full knowledge of the channel statistics and it is better than ALMMSE (an approximation to linear MMSE). The results confirm that this pipeline can be used efficiently in channel estimation.

# Datasets
links: 
Perfect channels - VehA model (without noise):
https://drive.google.com/file/d/1H5GiEWITfM00R4BS2uC3SiBLR0EZKX8m/view?usp=sharing

Noisy channels (SNR = 12dB);
https://drive.google.com/file/d/1mwnfXalDUTebreMZqUNHRGAENAeJL1Nn/view?usp=sharing

Noisy channels (SNR = 22dB);  
https://drive.google.com/file/d/1j0BcBoVKCDInryqfCRPjINAUrFrI_rxB/view?usp=sharing

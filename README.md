# Deep Learning-Based Channel Estimation
Implementation of the paper "Deep Learning-Based Channel Estimation" https://arxiv.org/abs/1810.05893

## Introduction
This repo contains a deep learning (DL) algorithm for channel estimation in communication systems. And the aim is to find the unknown values of the channel response using some known values at the pilot locations. 
The pipeline is based on [SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) and [DNCNN](https://arxiv.org/abs/1608.03981).

- Full paper PDF: [Deep Learning-Based Channel Estimation](https://arxiv.org/abs/1810.05893)
- Authors: Mehran Soltani, Vahid Pourahmadi, Ali Mirzaei, Hamid Sheikhzadeh

## Requirements and Dependencies
- cuda10.0 && cudnn7.6.5
- requirements.txt



## Datasets 

- Perfect channels - [VehA model](https://drive.google.com/file/d/1H5GiEWITfM00R4BS2uC3SiBLR0EZKX8m/view?usp=sharing) (without noise):
- Noisy channels [(SNR = 12dB)](https://drive.google.com/file/d/1mwnfXalDUTebreMZqUNHRGAENAeJL1Nn/view?usp=sharing)
- Noisy channels [(SNR = 22dB)](https://drive.google.com/file/d/1j0BcBoVKCDInryqfCRPjINAUrFrI_rxB/view?usp=sharing)

## BibTeX Citation
```
@article{soltani2019deep,
  title={Deep learning-based channel estimation},
  author={Soltani, Mehran and Pourahmadi, Vahid and Mirzaei, Ali and Sheikhzadeh, Hamid},
  journal={IEEE Communications Letters},
  volume={23},
  number={4},
  pages={652--655},
  year={2019},
  publisher={IEEE}
}
```

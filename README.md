% SNN Action Recognition

# Intro

This repository contains results for applying a Convolutional Neural Network (CNN) and a Spiking Neural Network (SNN) for learning action recognition on the UCF101 action dataset. This code takes advantage of the PyTorch framework for the CNN and the SNN uses Nengo_dl which incorporates deep learning into Nengo, a python library for simulating neural models.

# Installation and Usage
To install all of the libraries to run the code in this repository, do the following.

```
pip install -r requirements.txt
```

I HIGHLY recommend using a machine with a GPU when running this code. This made the total time for training about 10 epochs go from 200 minutes (with a AMD Ryzen Threadripper 2970WX 24-Core Processor) to 5 minutes (with a GTX 2080).

Also available is the data forward passed through the trained CNN. This and the pretrained weights of the SNN and CNN for this experiment are given here (TODO).



# Archeitecture

The CNN architecture is RESNET-252(TODO). The CNN used pretrained weights acquired from this repository (TODO). This CNN was trained on the same dataset, but also used a LSTM during training to learn temporal features. This archeitecture is popular and has shown success in the two-stream problem (DESCRIBE HERE).

The SNN architecture chosen for this project was based off of the Lagrange Memory Unit (LMU) (SOURCE) found here (TODO). This memory unit was chosen for two reasons. First it was shown that the LMU is better than the frequently used memory unit LSTM for longer lengths of time (SOURCE). This would be useful as this project will be applied to other data sources eventually; therefore, this would be a nice property to have for a dataset with large video segments during training. Second, code for the memory unit in Nengo was readily available online; however, it was given with non-spiking neurons, so some changes were made to make the network use spiking neurons.


# Nengo_dl

Documentation for nengo and nengo_dl can be found on their website (SOURCE) and in their respective papers (SOURCE,SOURCE).

# UCF101

This dataset was acquired from labeling and cutting youtube videos into 30 frame clips. There are 101 possible actions that each video can be labeled. This dataset was gathered from here (TODO).

# Results
TODO

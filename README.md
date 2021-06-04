# Deeplearning_termproject

## Contents
### Requirements
CUDA 10.2
Cudnn 8.0.5
Pytorch for cuda 10.2
matlab
numpy
opencv-python
efficient-net
* install latest versions for matlab, numpy and opencv-python
### - part1
1. tuner
 The purpose for the first set of codes is to make an automatic hyperparameter tuner. * base model used is Resnet18
 In this project we used 3 different loss functions(cross entropy loss, NLL loss, multi margin loss), 4 optimization functions(SGD, Adam, AdamW, RMSProp) and 5 differnt learning rates(1e-1, 1e-2, 1e-3, 1e-4, 1e-5).
2. how to run the code
#### About the code
 - config.py containes codes for running the tuner function. Each batch of experiments are done for 5 epochs.
 - config_optim.py contains codes for running different optimization functions.
 - config_loss.py contains codes for running different loss functions.
 - config_lr.py contains codes for running different lr.
 - config_settings.py contains codes for choosing different loss and optimization functions.
 - dataload.py contains codes for loading the data.
 - train.py contains codes for training validating and testing the model.
 - utils.py contains codes for saving a plot of models train, validation data loss and accuracy.
 #### How to run the code
 1. python config.py
 * this will get images from your image path and save weights along with the plot. Weight names have information of the lr, optimization function, loss function used with the results from the validation data.
### - part2
1. models used
- Densenet102 -> best acc 0.89
- Densenet201 -> training
2. how to run the code
#### About the code
- data_aug.py contains codes for augmenting the data.
- main.py contains codes for running the model.
- save_csv.py contains codes for saving the results of the test data to a csv file.
#### How to run the code
- if you want to augment your own data
-- python aug_data.py (this will augment and save your data in the orignial datafile)
- to run training sequence
-- python main.py
- to see accuracy results on the validation dataset
-- python test_data.py

# Fingerprint Pore Detection: A Survey 

## Project Description
This repository contains original implementation of ["Fingerprint Pore Detection: A survey"](). This repository contains implementation of 16 convolutional neural network architectures that were used during experiments described in the paper. Furthermore, this repository contains framework for researcher to conduct their own experiments and verify results described in the publication. 
Also, this repository contains unofficial re-implementations of methods described in ["Improving Fingerprint Pore Detection with a Small FCN"](https://arxiv.org/abs/1811.06846), ["A deep learning approach towards pore extraction for high-resolution fingerprint recognition"](https://ieeexplore.ieee.org/document/7952518), and ["Dynamic Pore Filtering for Keypoint Detection Applied to Newborn Authentication"](https://ieeexplore.ieee.org/document/6977010)

Additionally, this repository contains out-of-the-box code that allows people to detect pores with pre-trained models. 

## Outline
* [Requirements](#requirements)
* [Installation of dataset](#installation-of-dataset)
* [Experiment set up](#experiment-set-up)
* [Experiment result replication](#experiment-result-replication)
* [Detecting pores in arbitrary input images](#experiment-result-replication)
* [Research framework](#research-framework)
* [Re-implementations](#re-implementations)

## Requirements 
The code in this repository was tested for Ubuntu 20.04.3 and Python 3.8.10. Running experiments on this version will give results that are closest to what was reported in the research paper. 

To install required packages for this project, run the following command: 

```
pip install -r requirements.txt
```


## Installation of dataset

Like any other machine learning project, this one relies heavily on dataset usage. Thus, one needs to install a dataset before starting experiments. Any dataset that contains fingerprint images with annotated pore locations can be used. For our research experiments, either PolyU HREF or MS-RF datasets would work. 

The Hong Kong Polytechnic University High-Resolution-Fingerprint (PolyU HREF) is a **private** high-resolution fingerprint dataset for fingerprint recognition. This dataset contains annotated fingerprint pore location for 120 fingerprint images, and contains a subset of 30 partial fingerprint images. 

The L3 Synthetic Fingerprint (L3-SF) dataset is a **public** database of L3 synthetic fingerprint images for fingerprint recognition. This dataset contains fingerprint pore annotations for 740 images. 

To install the <ins>PolyU dataset</ins>, place a folder with the dataset inside of this repository. Then follow run the following command: 

```
chmod +x scripts/initializePolyU.sh
./scripts/initializePolyU.sh pathToDataset
```

To install the <ins>L3-SF dataset</ins>, place a folder with the dataset inside of this repository. Then follow run the following command: 

```
chmod +x scripts/initializeL3SF.sh
./scripts/initializeL3SF.sh pathToDataset
```

Note: You can install only one dataset at the same time. If you wish to switch the dataset, you need to delete the previous dataset and install the new dataset by using commands above.



## Experiment set up 
Before starting an experiment, ensure that you have created a directory where experiments results will be stored. The directory can be located anywhere, but it is preferable that it is located within experiments directory that is already preinstalled for you. The internal structure of the directory should be as follows: 

```
Experiment X/
    models/
    Prediction/
        Coordinates/
        Fingerprint/
        Pore/
```

Thus, PyTorch machine learning model files will be saved in ```Experiment X/models/ directory```. Text files with pore coordinates will be saved in ```Experiment X/Prediction/Coordinates/``` directory. Binary pore maps will be saved in ```Experiment X/Prediction/Pore/``` directory. Fingerprint images that showcase true and false detections will be saved in ```Experiment X/Prediction/Fingerprint/ directory```. 

To create the correct internal structure, you may use the following commands: 

```
chmod +x scripts/experimentSetUp.sh 
./scripts/experimetSetUp.sh NameOfTheExperiment
```


## Experiment result replication

If you are using <ins>PolyU</ins> Dataset, to replicate results of the experiments, run the following commands. 

```
chmod +x scripts/polyu13patchsize.sh
chmod +x scripts/polyu15patchsize.sh
chmod +x scripts/polyu17patchsize.sh
chmod +x scripts/polyu19patchsize.sh

./scripts/polyu13patchsize.sh  31-110 111-120 121-150 1-30
./scripts/polyu15patchsize.sh  31-110 111-120 121-150 1-30
./scripts/polyu17patchsize.sh  31-110 111-120 121-150 1-30
./scripts/polyu19patchsize.sh  31-110 111-120 121-150 1-30
```

where 31-110 denotes range of files that will be used for training, 111-120 denotes range of numbers that will be used for validation and 121-150 denotes range of numbers that will be used for first testing, and 1-30 for second testing. You may use a different protocol by changing the numbers.


If you are using <ins>L3-SF</ins>, to replicate results of the experiments, run the following commands. 

```
chmod +x scripts/l3sf13patchsize.sh
chmod +x scripts/l3sf15patchsize.sh
chmod +x scripts/l3sf17patchsize.sh
chmod +x scripts/l3sf19patchsize.sh

./scripts/l3sf13patchsize.sh  1-444 445-589 590-740
./scripts/l3sf15patchsize.sh  1-444 445-589 590-740
./scripts/l3sf17patchsize.sh  1-444 445-589 590-740
./scripts/l3sf19patchsize.sh  1-444 445-589 590-740
```

Experiment results will appear in experiments directory. To view the results of your experiments, you should locate an appropriate directory within experiments directory. 



## Detecting pores in arbitrary input images
If a user wishes to use a pretrained model, to detect pores on images inside of a dataset, then they may use this command below: 

```
chmod +x scripts/detect.sh 
./scripts/detect.sh 1-30
```

where 1-30 denotes for range of files that must be processed. The processed information will be saved in ```out_of_the_box_detect``` directory. 


## Research framework 

Additionally, this repository provides a framework to conduct pore detection research. Research can experiment with various command line arguments to tune hyperparameters to enhance results. Additionally, the framework provides a way for researchers to built their own CNN architectures and test them inside o this research environment: 

The following command line arguments are available, and researchers can tune them to get enhanced results: 

```
usage: train.py [-h] --patchSize PATCHSIZE [--poreRadius PORERADIUS] [--maxPooling MAXPOOLING] --experimentPath EXPERIMENTPATH
                [--trainingRange TRAININGRANGE] [--validationRange VALIDATIONRANGE] [--testingRange TESTINGRANGE]
                [--secondtestingRange SECONDTESTINGRANGE] --groundTruthFolder GROUNDTRUTHFOLDER [--residual]
                [--optimizer {Optimizer.ADAM,Optimizer.RMSPROP,Optimizer.SGD}] [--learningRate LEARNINGRATE]
                [--criteriation {Critireation.BCELOSS,Critireation.CROSSENTRAPY}] [--epoc EPOC] [--batchSize BATCHSIZE]
                [--device DEVICE] [--numberWorkers NUMBERWORKERS] [--testStartProbability TESTSTARTPROBABILITY]
                [--testEndProbability TESTENDPROBABILITY] [--testStepProbability TESTSTEPPROBABILITY]
                [--testStartNMSUnion TESTSTARTNMSUNION] [--testEndNMSUnion TESTENDNMSUNION] [--testStepNMSUnion TESTSTEPNMSUNION]
                [--numberFeatures NUMBERFEATURES] [--defaultNMS DEFAULTNMS] --boundingBoxSize BOUNDINGBOXSIZE
                [--defaultProb DEFAULTPROB] [--tolerance TOLERANCE] [--gabriel] [--su] [--softLabels]

optional arguments:
  -h, --help            show this help message and exit
  --patchSize PATCHSIZE
                        length of a patch that has a square shape. Thus, if a patch size is 17x17, enter 17
  --poreRadius PORERADIUS
                        radius arounda pore that are considered to be a pore in a ground truth data set
  --maxPooling MAXPOOLING
                        enables/disables maxpooling layers in the architecture
  --experimentPath EXPERIMENTPATH
                        directory where experiment information will be stored
  --trainingRange TRAININGRANGE
                        range of data set files that will be used for training
  --validationRange VALIDATIONRANGE
                        range of data set files that will be used for validation
  --testingRange TESTINGRANGE
                        range of data set files that will be used for testing
  --secondtestingRange SECONDTESTINGRANGE
                        range of data set files that will be used for second testing
  --groundTruthFolder GROUNDTRUTHFOLDER
                        Directory where the ground truth dataset is stored
  --residual            Enable/disable residual connections in the architecture
  --optimizer {Optimizer.ADAM,Optimizer.RMSPROP,Optimizer.SGD}
                        Optimizer that will be used during training
  --learningRate LEARNINGRATE
                        Learning rate that will be used during training
  --criteriation {Critireation.BCELOSS,Critireation.CROSSENTRAPY}
                        Criteriation that will be used during training
  --epoc EPOC           number of itterations during training
  --batchSize BATCHSIZE
                        size of a batch sample
  --device DEVICE       Device where training will be performed
  --numberWorkers NUMBERWORKERS
                        Number of GPU workers used during training
  --testStartProbability TESTSTARTPROBABILITY
                        Minimum probability threshold that will be considered when searching for optimal threshold
  --testEndProbability TESTENDPROBABILITY
                        Maximum probability threshold that will be considered when seaching for optimal threshold
  --testStepProbability TESTSTEPPROBABILITY
                        Step size when searching for optimal probability threshold
  --testStartNMSUnion TESTSTARTNMSUNION
                        Minimum NMS Union threshold that will be considered when searching for optimal threshold
  --testEndNMSUnion TESTENDNMSUNION
                        Maximum NMS Union threshold that will be considered when searching for optimal threshold
  --testStepNMSUnion TESTSTEPNMSUNION
                        Step size when searching for optimal NMS Union threshold
  --numberFeatures NUMBERFEATURES
                        number of features that the architecture uses in hidden layers
  --defaultNMS DEFAULTNMS
                        NMS Union values used during validation
  --boundingBoxSize BOUNDINGBOXSIZE
                        Size of bounding Box when performing NMS
  --defaultProb DEFAULTPROB
                        Probability values used during validation
  --tolerance TOLERANCE
                        Early stopping tolerance
  --gabriel             Unnoficial implementation of Gabriel Dahia's publication
  --su                  Unnoficial implementation of Su et. al. publication
  --softLabels          Use soft labels for annotations of pores
```

To run experiments with custom PyTorch CNN architectures, an architecture must be placed in the following location: 

```
architectures/
    yourPyTorchModel.py
    yourPyTorchModel2.py
    yourPytorchModel3.py
    .....
```

Here is an example on how one would use the research framework to run the basline described in the paper: 

```
python3 train.py --patchSize 17 --poreRadius 3 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3 --secondtestingRange $4 --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda  --softLabels >> experiments/17PatchSize/PolyUexp33.log;

```

## Re-implementations

### Gabriel Dahia (2018) FCN reimplementation model

In this paper, we have reimplemented FCN architecture developed by Gabriel Dahia in ["Improving Fingerprint Pore Detection with a Small FCN"](https://arxiv.org/abs/1811.06846). To run experiments using their architecture, run the following command:

```
chmod +x scripts/dahia.sh
./scripts/dahia.sh
```

### Su _et al._'s (2017) CNN reimplementation model
In this paper, we have reimplemented CNN architecture by Su et. al in ["A deep learning approach towards pore extraction for high-resolution fingerprint recognition"](https://ieeexplore.ieee.org/document/7952518). To run experiments using their architecture, run the following command:

```
chmod +x scripts/su.sh
./scripts/su.sh
```

### Lemes et. al (2014) Dynamic Pore Filtering - Computer Vision
In this paper, we have re-implemented dynamic pore filtering technique developed in ["Dynamic Pore Filtering for Keypoint Detection Applied to Newborn Authentication"](https://ieeexplore.ieee.org/document/6977010). To run experiments using their algorithm, run the following command: 

```
chmod +x scripts/dpf.sh
./scripts/dpf.sh
```

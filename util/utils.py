from cv2 import circle, imread, IMREAD_GRAYSCALE, resize
from psutil import WINDOWS
import torch
from tqdm import tqdm
from architectures import net15nomax, net15max, net17nomax, net17max, net19nomax, net19max, resnet15nomax, resnet15max, resnet17nomax, resnet17max, resnet19max, resnet19nomax, net13nomax, net13max, resnet13nomax,resnet13max
from argparse import ArgumentParser, ArgumentTypeError
import re
import architectures
from architectures.gabriel import Gabriel
from architectures.su import Su
import enum
from torch import nn


def plotPred(
        image, 
        pred, 
        radius, 
        color, 
        thickness, 
        WINDOW_SIZE, 
    ):

    for dim1 in range(len(pred[0])): 
        for dim2 in range(len(pred[0][0])): 
            if pred[0][dim1][dim2]: 
                circle(image, (dim2+WINDOW_SIZE//2, dim1+WINDOW_SIZE//2), radius=radius, color=color, thickness =  thickness)

    return image

def plotPredImage(
    image, 
    detectionList, 
    radius, 
    color, 
    thickness
): 
    for detection in detectionList: 
        circle(image, (detection[1], detection[0]), radius=radius, color=color, thickness =  thickness)

    return image


def getDimensions(image):
    return (image.shape[1], image.shape[2])


def plotGroundTruth(
    image, 
    filePath, 
    radius, 
    color, 
    thickness,
    
    ):
    with open(filePath) as f:
        lines = f.readlines()
        for number, i in enumerate(lines):
            i = i.replace(",", "")
            line = i.split()

            x = (int(line[0])-1) 
            y = (int(line[1])-1)
            circle(image, (y, x), radius=radius, color=color, thickness =  thickness,)

    return image 

def loadModel(modelPath, device, NUMBERLAYERS, NUMBERFEATURES, MAXPOOLING, WINDOWSIZE, residual, gabriel, su):  

  model = None

  if not residual:
    if gabriel: 
        model = Gabriel(NUMBERFEATURES)
    elif su: 
        model = Su(NUMBERFEATURES, 1)
    elif WINDOWSIZE == 13 and not MAXPOOLING: 
        model = net13nomax.Net13NoMax(NUMBERFEATURES)
    elif WINDOWSIZE == 13 and MAXPOOLING:
        model = net13max.Net13Max(NUMBERFEATURES) 
    elif WINDOWSIZE == 15 and not MAXPOOLING: 
        model = net15nomax.Net15NoMax(NUMBERFEATURES)
    elif WINDOWSIZE == 15 and MAXPOOLING:
        model = net15max.Net15Max(NUMBERFEATURES) 
    elif WINDOWSIZE == 17 and not MAXPOOLING: 
        model = net17nomax.Net17NoMax(NUMBERFEATURES)
    elif WINDOWSIZE == 17 and MAXPOOLING: 
        model = net17max.Net17Max(NUMBERFEATURES) 
    elif WINDOWSIZE == 19 and not MAXPOOLING: 
        model = net19nomax.Net19NoMax(NUMBERFEATURES)
    elif WINDOWSIZE == 19 and MAXPOOLING: 
        model = net19max.Net19Max(NUMBERFEATURES)
  else: 
    if WINDOWSIZE == 13 and not MAXPOOLING: 
        model = resnet13nomax.ResNet13NoMax(NUMBERFEATURES)
    elif WINDOWSIZE == 13 and MAXPOOLING:    
        model = resnet13max.ResNet13Max(NUMBERFEATURES)

    if WINDOWSIZE == 15 and not MAXPOOLING: 
        model = resnet15nomax.ResNet15NoMax(NUMBERFEATURES)
    elif WINDOWSIZE == 15 and MAXPOOLING:
        model = resnet15max.ResNet15Max(NUMBERFEATURES) 
    elif WINDOWSIZE == 17 and not MAXPOOLING: 
        model = resnet17nomax.ResNet17NoMax(NUMBERFEATURES)
    elif WINDOWSIZE == 17 and MAXPOOLING: 
        model = resnet17max.ResNet17Max(NUMBERFEATURES) 
    elif WINDOWSIZE == 19 and not MAXPOOLING: 
        model = resnet19nomax.ResNet19NoMax(NUMBERFEATURES)
    elif WINDOWSIZE == 19 and MAXPOOLING: 
        model = resnet19max.ResNet19Max(NUMBERFEATURES)


  model.load_state_dict(torch.load(modelPath, map_location=torch.device(device)))
  return model

def writeCoordinates(
    filePath, 
    pred, 
    WINDOW_SIZE, 
    ): 
    with open(filePath, mode="w") as f:
        for dim1 in range(pred.shape[0]): 
            for dim2 in range(pred.shape[1]): 
                if pred[dim1][dim2]: 
                    f.writelines(str((dim1+WINDOW_SIZE//2)) + ", " + str((dim2+WINDOW_SIZE//2)) + "\n")


def testSet(imagePath, labelPath):
    print(imagePath)
    print(labelPath)
    image = imread(imagePath, IMREAD_GRAYSCALE) 
    image = plotGroundTruth(image, labelPath , 3, [255, 0, 0], 1)

    return image



def train_loop(dataloader, model, loss_fn, optimizer, batchSize, device, gabriel, lr_shedule):
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.type(torch.float32).to(device), y.type(torch.float32).to(device)
        
        pred = model(X)
        pred = torch.flatten(pred)
        pred=pred.type(torch.float32)
        y=y.type(torch.float32)

        loss = loss_fn(pred, y)

        pred, loss = pred.to(device), loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if gabriel: 
            if batch*batchSize % 2000 == 0:
                lr_shedule.step()


        if batch == 1563: 
            break



def parseNumList(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))


def rangeNotContinuous(string):
    if "," in string: 
        split = string.split(",")
        first = parseNumList(split[0])
        second = parseNumList(split[1])
        return list(first) + list(second)        
        

    else: 
        return parseNumList(string)


class Macro: 
    def __init__(self) -> None:
        pass


class Optimizer(enum.Enum): 
    ADAM = "ADAM"
    RMSPROP = "RMSPROP"
    SGD = "SGD"


class Critireation(enum.Enum): 
    BCELOSS = "BCELOSS"
    CROSSENTRAPY = "CROSSENTRAPY"
     







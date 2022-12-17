import numpy, torch, argparse
from util.utils import loadModel as lm 
import entireImage, cv2, torchvision, copy, multiprocessing
from validate import readTxtList
from util.utils import plotPredImage as draw
from util.utils import parseNumList
from tqdm import tqdm
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from torch import nn


def nms_wrapper(args):
   return nms(*args)

def nms(pred, fileIndex, validationImagesPaths): 
    entireImage.apply_nms(pred, 0.65, 17, 0.2, "out_of_the_box_detect/Prediction/Pore/", 1+fileIndex, "out_of_the_box_detect/Prediction/Coordinates/", 17)
    pred = pred.narrow(2, 16, pred[0].size(1) - 2*16).narrow(3, 16, pred[0].size(1) - 2*16)
    detections = []
    currentImage = cv2.imread(validationImagesPaths[fileIndex], cv2.IMREAD_GRAYSCALE)
    readTxtList(
    "out_of_the_box_detect/Prediction/Coordinates/" + "%d.txt" % (fileIndex+1), 
    detections, 
    currentImage.shape[0], 
    currentImage.shape[1], 
    17, 
    )

    image = draw(currentImage, detections, 5, [0, 0, 255], 1)

    cv2.imwrite("out_of_the_box_detect/Prediction/Fingerprint/%d.png" % (fileIndex+1), image)


def inference_wrapper(args): 
    return inference(*args)

def inference(model, image, predictedImages, transforms, device):
    return model(transforms(image).unsqueeze(dim=0).float().to(device)).detach().cpu()



if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    multiprocessing.set_start_method("spawn")


    parser.add_argument('--groundTruthFolder', 
                    required=True,
                    type=str, 
                    help="Directory where the ground truth dataset is stored"
                    )  

    parser.add_argument('--testingRange', 
                    default=None,
                    type=parseNumList, 
                    help="range of data set files that will be used for testing"
                    )    

    parser.add_argument('--features', 
                    default=64,
                    type=int, 
                    help="Number of features of the NN architecture"
                    )    

    parser.add_argument('--device', 
                    default="cuda",
                    type=str, 
                    help="Device: either cuda or cpu"
                    )    
                


    args = parser.parse_args()


    pathToSolution = "out_of_the_box_detect/"
    test_range = list(args.testingRange)
    GROUNDTRUTH = args.groundTruthFolder

    model = lm(modelPath=pathToSolution+f"models/{args.features}", 
    device=torch.device(args.device), 
    NUMBERLAYERS=8, 
    NUMBERFEATURES=int(args.features), 
    MAXPOOLING=False, 
    WINDOWSIZE=17, 
    residual=False, 
    gabriel=False, 
    su=False)
    

    model.eval()
    model.to(args.device)


    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

    validationImagesPaths = [GROUNDTRUTH + "/PoreGroundTruthSampleimage/" + "%s.bmp" % fileIndex for fileIndex in test_range]
    validationImages = numpy.array([cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in validationImagesPaths])

    predictedImages = [] #torch.zeros([len(validationImages), 1, 224, 304])

    modelargs = []
    for image in validationImages: 
        modelargs.append([model, image, predictedImages, transforms, args.device])

    predictedImages = process_map(inference_wrapper, modelargs, max_workers=1)

    poolargs = []
    poolpred, poolindex, poolpath = [], [], []
    for fileIndex, pred in enumerate(predictedImages):
        poolargs.append([pred, fileIndex, validationImagesPaths])


    process_map(nms_wrapper, poolargs, max_workers=4)



        

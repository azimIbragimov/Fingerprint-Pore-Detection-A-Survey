import numpy, torch, argparse
from util.utils import loadModel as lm 
import entireImage, cv2, torchvision, copy, multiprocessing
from validate import readTxtList
from util.utils import plotPredImage as draw
from util.utils import parseNumList
from tqdm import tqdm

if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()


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

    args = parser.parse_args()

    pathToSolution = "out_of_the_box_detect/"
    test_range = list(args.testingRange)
    GROUNDTRUTH = args.groundTruthFolder

    model = lm(modelPath=pathToSolution+"models/Best", 
    device=torch.device("cpu"), 
    NUMBERLAYERS=8, 
    NUMBERFEATURES=64, 
    MAXPOOLING=False, 
    WINDOWSIZE=17, 
    residual=False, 
    gabriel=False, 
    su=False)

    model.eval()
    model.to("cpu")

    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

    validationImagesPaths = [GROUNDTRUTH + "/PoreGroundTruthSampleimage/" + "%s.bmp" % fileIndex for fileIndex in test_range]
    validationImages = numpy.array([cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in validationImagesPaths])

    predictedImages = []  
    for image in tqdm(validationImages): 
        tensorImage = transforms(image)
        predictionImage = model(tensorImage.float().unsqueeze(dim=0).to("cpu")).detach().cpu()
        predictedImages.append(predictionImage)

    for fileIndex, pred in (enumerate(predictedImages)):
        entireImage.apply_nms(pred, 0.6, 17, 0.45, "out_of_the_box_detect/Prediction/Pore/", 1+fileIndex, "out_of_the_box_detect/Prediction/Coordinates/", 17)
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
        
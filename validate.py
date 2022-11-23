# %%
from scipy.spatial import distance
import numpy, cv2, os
from scipy.spatial import distance_matrix


from util.utils import plotPred, plotPredImage

class Pore: 

    def __init__(self, x, y):
        self.x = x 
        self.y = y 

    def getList(self): 
        return [self.x, self.y]

    def __str__(self) -> str:
        
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"



def getDistance(pore1, pore2): 
    return abs(distance.euclidean(pore1.getList(), pore2.getList()))



def readTxt(path, storage, imageXDimension, imageYDimension, windowsSize):
    with open(path) as f:
        lines = f.readlines()
        for number, i in enumerate(lines):
            i = i.replace(",", "")
            line = i.split()
            if not line[0].isdigit() or not line[1].isdigit(): continue
            x = (int(line[0])-1)
            y = (int(line[1])-1)

            half = windowsSize//2

            if half < x < imageXDimension-half and half < y < imageYDimension-half:
                storage.append(Pore(x, y))


def readTxtList(path, storage, imageXDimension, imageYDimension, windowsSize):
    with open(path) as f:
        lines = f.readlines()
        for number, i in enumerate(lines):
            i = i.replace(",", "")
            line = i.split()
            if not line[0].isdigit() or not line[1].isdigit(): continue
            x = (int(line[0])-1)
            y = (int(line[1])-1)

            half = windowsSize//2

            if half < x < imageXDimension-half and half < y < imageYDimension-half:
                storage.append([x, y])

def test(
        index, 
        groundTruthCoordinatesFolder, 
        predictionCoordinatesFolder, 
        imageXDimension, 
        imageYDimension, 
        windowsSize,
        firstIndex, 
        stats
        ):


    groundTruth, predictions = [], []
    groundTruthTest, predictionsTest = [], []


    files = [os.path.isfile(groundTruthCoordinatesFolder + "%d.txt" % index)]
    if False in files: 
        return [0, 0, 0, 0]

    originalImage = cv2.imread(groundTruthCoordinatesFolder + "../PoreGroundTruthSampleimage/" + "%s.bmp" % index)

    imageXDimension = originalImage.shape[0]
    imageYDimension = originalImage.shape[1]

    readTxt(
        groundTruthCoordinatesFolder + "%d.txt" % index, 
        groundTruth, 
        imageXDimension, 
        imageYDimension, 
        windowsSize, 
        )

    readTxt(
        predictionCoordinatesFolder + "%d.txt" % index, 
        predictions, 
        imageXDimension, 
        imageYDimension, 
        windowsSize, 
        )

    readTxtList(
        groundTruthCoordinatesFolder + "%d.txt" % index, 
        groundTruthTest, 
        imageXDimension, 
        imageYDimension, 
        windowsSize, 
        )

    readTxtList(
        predictionCoordinatesFolder + "%d.txt" % index, 
        predictionsTest, 
        imageXDimension, 
        imageYDimension, 
        windowsSize, 
        )



    groundTruthTest, predictionsTest = numpy.array(groundTruthTest), numpy.array(predictionsTest)
    groundTruth, predictions = numpy.array(groundTruth, dtype=Pore), numpy.array(predictions, dtype=Pore)

    if len(groundTruthTest) == 0 or len(predictionsTest) == 0: 
        stats.put([0, 0, 0, 0])
        return  

    totalPred = len(predictions)
    totalGT = len(groundTruth)
    hs1, hs2 = dict(), dict()

    hstest1, hstest2 = dict(), dict()

    for g in numpy.arange(groundTruthTest.shape[0]): 
        matrix = distance_matrix(predictionsTest, [groundTruthTest[g]])
        correspondence = predictionsTest[numpy.argmin(matrix)]
        hstest1[tuple(groundTruthTest[g])] = tuple(correspondence)


    for p in numpy.arange(predictionsTest.shape[0]): 
        matrix = distance_matrix(groundTruthTest, [predictionsTest[p]])
        correspondence = groundTruthTest[numpy.argmin(matrix)]
        hstest2[tuple(predictionsTest[p])] = tuple(correspondence)



    hs1, hs2 = hstest1, hstest2


    # for g in numpy.arange(len(groundTruth)): 
    #     minDistnace = float("inf")
    #     minPore = None

    #     # start, end = g.copy() * 255,  groundTruth.copy()
    #     # print(start, end)



    #     for p in numpy.arange(len(predictions)): 
    #         if minDistnace > getDistance(groundTruth[g], predictions[p]): 
    #             minDistnace = getDistance(groundTruth[g], predictions[p])
    #             minPore = predictions[p] 

    #         hs1[groundTruth[g]] = minPore 


    # falsePositives = predictions.copy()


    # for p in numpy.arange(len(predictions)): 
    #     minDistnace = float("inf")
    #     minPore = None
    #     for g in numpy.arange(len(groundTruth)): 
    #         if minDistnace > getDistance(groundTruth[g], predictions[p]): 
    #             minDistnace = getDistance(groundTruth[g], predictions[p])
    #             minPore = groundTruth[g] 

    #         hs2[predictions[p]] = minPore
    #         # falsePositives.remove(minPore)

    # print(hs2)
    trueDetections, falseDetections = [], [] 

    for key, item in hs1.items(): 
        if key == hs2[item]: 
            trueDetections.append(key)
        else: 
            falseDetections.append(key)



    originalImage = cv2.imread(groundTruthCoordinatesFolder + "../PoreGroundTruthSampleimage/%s.bmp" % index)
    listTrueDetection = [prediction for prediction in trueDetections]
    listFalseDetection = [prediction for prediction in falseDetections]
    # falsePositivesDetection = [prediction for prediction in falsePositives]

    image = plotPredImage(originalImage, listTrueDetection, 3, [0, 255, 0], 1)
    image = plotPredImage(originalImage, listFalseDetection, 3, [0, 0, 255], 1)
    # image = plotPredImage(originalImage, falsePositivesDetection, 3, [0, 255, 0], 1)


    cv2.imwrite(predictionCoordinatesFolder + "../Fingerprint/%d.png" % index, image)

    stats.put([len(trueDetections), len(falseDetections), len(predictions), totalGT])





import torch, cv2, torchvision
from util.utils import plotPred, getDimensions, plotGroundTruth, loadModel, writeCoordinates
from tqdm import tqdm
import os 
from itertools import repeat
from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing as mp



# function that takes prediction of ML model and its plots results and saves in a txt file
def findPores(
        model, 
        index, 
        probability, 
        imageFolder,
        labelFolder, 
        fingerprintPredFolder, 
        porePredFolder, 
        coordinatePredFolder,
        nsmThreshold, 
        nmsWindow, 
        device, 
        NUMBERLAYERS, 
        NUMBERFEATURES,
        MAXPOOLING, 
        residual, 
        gabriel, 
        su, 
        boundingBoxSize, 

        preDefinedPrediction = None

    ):

    if preDefinedPrediction == None:

        # load the model on CPU
        model = loadModel(
            modelPath = model, 
            device = device, 
            NUMBERLAYERS = NUMBERLAYERS, 
            NUMBERFEATURES = NUMBERFEATURES,
            MAXPOOLING = MAXPOOLING, 
            WINDOWSIZE=nmsWindow, 
            residual=residual, 
            gabriel=gabriel, 
            su=su 
            )
        # turn on the testing mode
        model.eval()
        model.to(device)
        # go through each file

        torch.backends.cudnn.benchmark = True


        files = [os.path.isfile(imageFolder + "%s.bmp" % fileIndex) for fileIndex in index]
        if False in files: 
            return

            ans1, ans2 = 0, 0 
    processes = []


    for fileIndex in (index):

        if preDefinedPrediction == None:

            if not os.path.isfile(imageFolder + "%s.bmp" % fileIndex): 
                break

            # read image and transoform it into proper tensor
            image = cv2.imread(imageFolder + "%s.bmp" % fileIndex, cv2.IMREAD_GRAYSCALE)
            
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),        ])
            image = transforms(image).to(device)
            y, x = getDimensions(image)
            image = image.reshape(1, 1, y, x)

            # get the prediction
            with torch.no_grad():
                pred = model(image.clone().detach().to(device)).cpu()

        else: 
            pred = preDefinedPrediction[fileIndex-index[0]]

        p = mp.Process(target=apply_nms, args=(
                pred, 
                probability, 
                boundingBoxSize, 
                nsmThreshold, 
                porePredFolder, 
                fileIndex, 
                coordinatePredFolder, 
                nmsWindow, 
                ))

        p.start()
        processes.append(p)

    for p in (processes):
        p.join()



def apply_nms(
    pred, 
    probability, 
    boundingBoxSize, 
    nsmThreshold, 
    porePredFolder, 
    fileIndex, 
    coordinatePredFolder, 
    nmsWindow, 
    drawPredictions = False, 
    device="cpu"
    ):

        #apply nms
        workpred = pred.squeeze()
        boxes = torch.tensor([], device=device)
        scores = torch.tensor([], device=device)

        mask = torch.ones(workpred.shape, dtype=torch.int16)
        sim_vec = torch.nonzero((workpred >= probability)*mask)
        sim_vec2 = sim_vec + boundingBoxSize
        cat = torch.cat((sim_vec, sim_vec2), dim=1)

        coordinates = cat[:, :2]

        scorescat = torch.zeros(coordinates.shape[0])
        for i, coordinate in enumerate(coordinates): 
            scorescat[i] = workpred[coordinate[0], coordinate[1]]



        boxes, score = cat, scorescat


        indices = torchvision.ops.boxes.nms(boxes.type(torch.float), score.type(torch.float), nsmThreshold)

        pred[0][0] = torch.zeros(pred[0][0].shape)


        for i in indices: 
            x1, y1, x2, y2 = boxes[i]
            center = (x1, y1)
            pred[0][0][int(center[0])][int(center[1])] = 1


        # pred = (pred>0.95).float()
        pred = pred.squeeze()
        pred = pred.detach().numpy()
        
        cv2.imwrite(porePredFolder + "%d.png" % fileIndex, pred*255)


        writeCoordinates(coordinatePredFolder + "%d.txt" % fileIndex, pred * 255, 
        WINDOW_SIZE = nmsWindow,
        )
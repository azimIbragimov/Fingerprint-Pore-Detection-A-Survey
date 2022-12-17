import torch, numpy, cv2, torchvision
from datasetPores import datasetPores
from util.utils import Optimizer, Critireation
import validate, entireImage, util
import entireImage, validate, numpy 
import datetime
from architectures import net15nomax, net15max, net17nomax, net17max, net19nomax, net19max, resnet15nomax, resnet15max, resnet17nomax, resnet17max, resnet19max, resnet19nomax, gabriel, su, net13max, net13nomax, resnet13max, resnet13nomax
import argparse, os
from tqdm import tqdm
import torch.multiprocessing as mp
import time
import copy, random

# setting the seed to exlcude non-deterministic behaviour 
torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)
torch.cuda.manual_seed(0)


parser = argparse.ArgumentParser()

parser.add_argument('--patchSize', 
                    required=True, 
                    type=int,
                    help="length of a patch that has a square shape. Thus, if a patch size is 17x17, enter 17"
                    )


parser.add_argument('--poreRadius', 
                    default=5,
                    type=int, 
                    help="radius arounda pore that are considered to be a pore in a ground truth data set"
                    )

parser.add_argument('--maxPooling', 
                    default=False,
                    type=bool,
                    help="enables/disables maxpooling layers in the architecture"
                    )

parser.add_argument('--experimentPath', 
                    required=True,
                    type=str,
                    help="directory where experiment information will be stored"
                    )

parser.add_argument('--trainingRange', 
                    default=None,
                    type=util.utils.rangeNotContinuous, 
                    help="range of data set files that will be used for training"
                    )    

parser.add_argument('--validationRange', 
                    default=None,
                    type=util.utils.parseNumList, 
                    help="range of data set files that will be used for validation"
                    )       

parser.add_argument('--testingRange', 
                    default=None,
                    type=util.utils.parseNumList, 
                    help="range of data set files that will be used for testing"
                    )    

parser.add_argument('--secondtestingRange', 
                    default=None,
                    type=util.utils.parseNumList, 
                    help="range of data set files that will be used for second testing"
                    )    

parser.add_argument('--groundTruthFolder', 
                    required=True,
                    type=str, 
                    help="Directory where the ground truth dataset is stored"
                    )  

parser.add_argument('--residual', 
                    default=False, 
                    action='store_true',
                    help="Enable/disable residual connections in the architecture"
                    )

parser.add_argument("--optimizer", 
                    default=util.utils.Optimizer.ADAM,
                    type=util.utils.Optimizer, 
                    choices=list(util.utils.Optimizer),
                    help="Optimizer that will be used during training"
                    )

parser.add_argument("--learningRate", 
                    default=3e-5,
                    type=float, 
                    help="Learning rate that will be used during training"
                    )

parser.add_argument("--criteriation", 
                    default=Critireation.BCELOSS,
                    type=Critireation, 
                    choices=list(Critireation),
                    help="Criteriation that will be used during training"
                    )

parser.add_argument('--epoc', 
                    default=10, 
                    type=int, 
                    help='number of itterations during training'
                    )

parser.add_argument('--batchSize', 
                    default=128, 
                    type=int, 
                    help = 'size of a batch sample'
                    )

parser.add_argument('--device', 
                    default="cuda:0", 
                    type=str, 
                    help="Device where training will be performed"
                    )


parser.add_argument('--numberWorkers', 
                    default=1,
                    type=int, 
                    help="Number of GPU workers used during training"
                    )


parser.add_argument('--testStartProbability', 
                    default=0,
                    type=float, 
                    help="Minimum probability threshold that will be considered when searching for optimal threshold"
                    )  

parser.add_argument('--testEndProbability', 
                    default=1,
                    type=float, 
                    help="Maximum probability threshold that will be considered when seaching for optimal threshold"
                    )  

parser.add_argument('--testStepProbability', 
                    default=0.05,
                    type=float, 
                    help="Step size when searching for optimal probability threshold"
                    )  

parser.add_argument('--testStartNMSUnion', 
                    default=0,
                    type=float, 
                    help="Minimum NMS Union threshold that will be considered when searching for optimal threshold"
                    )  

parser.add_argument('--testEndNMSUnion', 
                    default=1,
                    type=float, 
                    help="Maximum NMS Union threshold that will be considered when searching for optimal threshold"
                    )  

parser.add_argument('--testStepNMSUnion', 
                    default=0.05,
                    type=float, 
                    help="Step size when searching for optimal NMS Union threshold"
                    )  

parser.add_argument('--numberFeatures', 
                    default=64,
                    type=int, 
                    help='number of features that the architecture uses in hidden layers'
                    )  


parser.add_argument('--defaultNMS', 
                    default=0.20,
                    type=float, 
                    help="NMS Union values used during validation"
                    )  

parser.add_argument('--boundingBoxSize', 
                    required=True, 
                    type=int, 
                    help="Size of bounding Box when performing NMS"
                    )

parser.add_argument('--defaultProb', 
                    default=0.5,
                    type=float, 
                    help="Probability values used during validation"
                    )  

parser.add_argument('--tolerance', 
                    default=2,
                    type=int, 
                    help="Early stopping tolerance"
                    )  

parser.add_argument('--gabriel', 
                    action='store_true',
                    help="Unnoficial implementation of Gabriel Dahia's publication"
                    )

parser.add_argument('--su', 
                    default=False, 
                    action='store_true',
                    help="Unnoficial implementation of Su et. al. publication"
                    )

parser.add_argument('--softLabels', 
                    default=False, 
                    action='store_true',
                    help="Use soft labels for annotations of pores"
                    )


args = parser.parse_args()


def algorithm():

  """
  This function finds pores in a fingerprint
  """
  torch.backends.cudnn.benchmark = False
  
  MACRO = util.utils.Macro
  MACRO.CRITERIA = "BCELOSS"
  MACRO.OPTIMIZER = "ADAM"

  # macrodefinitions

  MACRO.EPOC = args.epoc
  MACRO.DEVICE = args.device
  MACRO.WINDOW_SIZE = args.patchSize
  MACRO.TRAIN_RANGE = args.trainingRange
  MACRO.VALIDATION_RANGE = args.validationRange
  MACRO.TEST_RANGE = args.testingRange
  MACRO.NMS_THRESHOLD = args.defaultNMS
  MACRO.DEFAULT_PROBABILITY = args.defaultProb
  MACRO.TOLERANCE = args.tolerance
  MACRO.PORELABELRADIUS = args.poreRadius
  MACRO.MAXPOOLING = args.maxPooling
  MACRO.BATCH_SIZE = args.batchSize
  MACRO.MODELPATH = args.experimentPath + "/models/Imbalanced"
  MACRO.BESTMODELPATH = args.experimentPath + "/models/Best"
  MACRO.PREDICTIONFINGERPRINT = args.experimentPath + f"/Prediction/Fingerprint/"
  MACRO.PREDICTIONPORE = args.experimentPath + f"/Prediction/Pore/"
  MACRO.PREDICTIONCOORDINATES = args.experimentPath + "/Prediction/Coordinates/"
  MACRO.NUM_WORKERS = args.numberWorkers
  MACRO.TEST_START_PROBABILITY = args.testStartProbability    # Minimum threshold probability that will be checked 
  MACRO.TEST_END_PROBABILITY = args.testEndProbability        # Maximum threshold probability that will be checked
  MACRO.TEST_STEP = args.testStepProbability                # Step while testing threshold probabilities
  MACRO.TEST_START_NMS = args.testStartNMSUnion              # Minimum NMS threshold probability that will be checked  
  MACRO.TEST_END_NMS = args.testEndNMSUnion              # Maximum NMS threshold probability that will be checked
  MACRO.STEP_NMS = args.testStepNMSUnion               # Step while testing NMS threshold probabilities
  MACRO.NUMBERFEATURES = args.numberFeatures
  MACRO.GROUNDTRUTHIMAGES = args.groundTruthFolder + "/PoreGroundTruthSampleimage/"
  MACRO.GROUNDTRUTHLABELS = args.groundTruthFolder + "/PoreGroundTruthMarked/"



  device = torch.device(MACRO.DEVICE)

  print(args.residual)
  
  if args.residual == False:
    if args.gabriel: 
        print("Running experiment of Gabriel Dahia")
        model = gabriel.Gabriel(MACRO.NUMBERFEATURES)
    elif args.su: 
        print("Running experiment of Su et. al")
        model = su.Su(MACRO.NUMBERFEATURES, MACRO.BATCH_SIZE)
    elif MACRO.WINDOW_SIZE == 13 and not MACRO.MAXPOOLING: 
        model = net13nomax.Net13NoMax(MACRO.NUMBERFEATURES)
    elif MACRO.WINDOW_SIZE == 13 and MACRO.MAXPOOLING:
        model = net13max.Net13Max(MACRO.NUMBERFEATURES)         
    elif MACRO.WINDOW_SIZE == 15 and not MACRO.MAXPOOLING: 
        model = net15nomax.Net15NoMax(MACRO.NUMBERFEATURES)
    elif MACRO.WINDOW_SIZE == 15 and MACRO.MAXPOOLING:
        model = net15max.Net15Max(MACRO.NUMBERFEATURES) 
    elif MACRO.WINDOW_SIZE == 17 and not MACRO.MAXPOOLING: 
        model = net17nomax.Net17NoMax(MACRO.NUMBERFEATURES)
    elif MACRO.WINDOW_SIZE == 17 and MACRO.MAXPOOLING: 
        model = net17max.Net17Max(MACRO.NUMBERFEATURES) 
    elif MACRO.WINDOW_SIZE == 19 and not MACRO.MAXPOOLING: 
        model = net19nomax.Net19NoMax(MACRO.NUMBERFEATURES)
    elif MACRO.WINDOW_SIZE == 19 and MACRO.MAXPOOLING: 
        model = net19max.Net19Max(MACRO.NUMBERFEATURES)
    else: 
        print("Model not defined. Aborting the experiment")
        return 0  
  else: 
    if MACRO.WINDOW_SIZE == 13 and not MACRO.MAXPOOLING: 
        model = resnet13nomax.ResNet13NoMax(MACRO.NUMBERFEATURES)
    elif MACRO.WINDOW_SIZE == 13 and MACRO.MAXPOOLING:
        model = resnet13max.ResNet13Max(MACRO.NUMBERFEATURES) 
    elif MACRO.WINDOW_SIZE == 15 and not MACRO.MAXPOOLING: 
        model = resnet15nomax.ResNet15NoMax(MACRO.NUMBERFEATURES)
    elif MACRO.WINDOW_SIZE == 15 and MACRO.MAXPOOLING:
        model = resnet15max.ResNet15Max(MACRO.NUMBERFEATURES) 
    elif MACRO.WINDOW_SIZE == 17 and not MACRO.MAXPOOLING: 
        model = resnet17nomax.ResNet17NoMax(MACRO.NUMBERFEATURES)
    elif MACRO.WINDOW_SIZE == 17 and MACRO.MAXPOOLING: 
        model = resnet17max.ResNet17Max(MACRO.NUMBERFEATURES) 
    elif MACRO.WINDOW_SIZE == 19 and not MACRO.MAXPOOLING: 
        model = resnet19nomax.ResNet19NoMax(MACRO.NUMBERFEATURES)
    elif MACRO.WINDOW_SIZE == 19 and MACRO.MAXPOOLING: 
        model = resnet19max.ResNet19Max(MACRO.NUMBERFEATURES)
    else: 
        print("Model not defined. Aborting the experiment")
        return 0  

  MACRO.NUMBERLAYERS = model.numberLayers
  model = model.to(device)

  batchSize = MACRO.BATCH_SIZE

  transforms = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.RandomHorizontalFlip(), 
      torchvision.transforms.RandomVerticalFlip(), 
      torchvision.transforms.ColorJitter(brightness=(0, 1), contrast=(0, 1))
  ])
  best_prob, best_NMS = None, None

  if MACRO.TRAIN_RANGE:  
    train_set = datasetPores(
        train = True, 
        transform=transforms,
        TRAIN_RANGE = MACRO.TRAIN_RANGE,
        GROUNDTRUTHIMAGES = MACRO.GROUNDTRUTHIMAGES,
        GROUNDTRUTHLABELS = MACRO.GROUNDTRUTHLABELS, 
        PORELABELRADIUS = MACRO.PORELABELRADIUS, 
        WINDOW_SIZE = MACRO.WINDOW_SIZE, 
        softLabels=args.softLabels
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, 
        batch_size=batchSize, 
        num_workers=MACRO.NUM_WORKERS, 
        shuffle=True, 
        pin_memory=True
        )


  if args.criteriation == util.utils.Critireation.BCELOSS: 
    criterion = torch.nn.BCELoss()
  elif args.criteriation == util.utils.Critireation.CROSSENTRAPY:
    criterion = torch.nn.CrossEntropyLoss()

  lr_scheduler = None  
  if args.optimizer == Optimizer.ADAM: 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate) 
  elif args.optimizer == Optimizer.RMSPROP:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learningRate)
  elif args.optimizer == Optimizer.SGD: 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learningRate)    
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96, last_epoch=-1)  

  else: 
    print("Optimizer not provided")
    print(args.optimizer)
    return  

  #Printing information about the current expiriment: 
  print("-----------------------------------------")
  print("Information about the current expreriment")
  print("-----------------------------------------")
  if args.residual: 
        print( "{: <40} {: >60}".format("Type", "Residual"))
  print( "{: <40} {: >60}".format("Start of the experiment:", datetime.datetime.now()))
  print( "{: <40} {: >60}".format("Number of epocs:", MACRO.EPOC))
  print( "{: <40} {: >60}".format("Device:", MACRO.DEVICE))
  print( "{: <40} {: >60}".format("Batch Size:", MACRO.BATCH_SIZE))
  print( "{: <40} {: >60}".format("Number of Workers:", MACRO.NUM_WORKERS))
  print( "{: <40} {: >60}".format("Default NMS Threshold:", MACRO.NMS_THRESHOLD))
  print( "{: <40} {: >60}".format("Windows Size:", MACRO.WINDOW_SIZE))
  print( "{: <40} {: >60}".format("Train Range:", str(MACRO.TRAIN_RANGE)))
  print( "{: <40} {: >60}".format("Validation Range:", str(MACRO.VALIDATION_RANGE)))
  print( "{: <40} {: >60}".format("Test Range:", str(MACRO.TEST_RANGE)))
  print( "{: <40} {: >60}".format("Latest Model Path:", MACRO.MODELPATH))
  print( "{: <40} {: >60}".format("Best Model Path:", MACRO.BESTMODELPATH))
  print( "{: <40} {: >60}".format("Default Probability:", MACRO.DEFAULT_PROBABILITY))
  print( "{: <40} {: >60}".format("Ground Truth Image Directory", MACRO.GROUNDTRUTHIMAGES))
  print( "{: <40} {: >60}".format("Ground Truth Image Directory", MACRO.GROUNDTRUTHIMAGES))
  print( "{: <40} {: >60}".format("Ground Truth Labels Directory", MACRO.GROUNDTRUTHLABELS))
  print( "{: <40} {: >60}".format("Prediction Pore Map Directory", MACRO.PREDICTIONPORE))
  print( "{: <40} {: >60}".format("Prediction Coordinate Directory", MACRO.PREDICTIONCOORDINATES))
  print( "{: <40} {: >60}".format("Pore Label Radius:", MACRO.PORELABELRADIUS))
  print( "{: <40} {: >60}".format("Architecture Maxpooling:", MACRO.MAXPOOLING))
  print("Critera:", criterion)
  print("Optimizer:", optimizer)
  print("Architecture:", model)


  # Start Training
  print("-----------------------------------------")
  print("Training & Validation")

  maxResult = 0
  if MACRO.TRAIN_RANGE:
    for i in range(MACRO.EPOC):
        print(f"\nEpoc", i)
        print("---------------------------------------")
        print("Time: ", datetime.datetime.now())
        util.utils.train_loop(train_loader, model, criterion, optimizer, batchSize, device, args.gabriel, lr_scheduler) #train the model
        torch.save(model.state_dict(), MACRO.MODELPATH) # save the model

        if MACRO.VALIDATION_RANGE: 
            
            #multiprocessing for validation 
            processes = []
            entireImage.findPores(MACRO.MODELPATH, 
                MACRO.VALIDATION_RANGE,   #inside of the validation range
                MACRO.DEFAULT_PROBABILITY, 
                MACRO.GROUNDTRUTHIMAGES, 
                MACRO.GROUNDTRUTHLABELS,
                MACRO.PREDICTIONFINGERPRINT, 
                MACRO.PREDICTIONPORE, 
                MACRO.PREDICTIONCOORDINATES, 
                MACRO.NMS_THRESHOLD, 
                MACRO.WINDOW_SIZE,
                torch.device(MACRO.DEVICE), 
                MACRO.NUMBERLAYERS, 
                MACRO.NUMBERFEATURES,
                MACRO.MAXPOOLING, 
                args.residual, 
                args.gabriel, 
                args.su, 
                args.boundingBoxSize)

            # add F-score here
            queue = mp.Queue()
            processes = []
            stats = []
            for i in MACRO.VALIDATION_RANGE: 
                p = mp.Process(target=  validate.test, args=(i, MACRO.GROUNDTRUTHLABELS, MACRO.PREDICTIONCOORDINATES, -1, -1, MACRO.WINDOW_SIZE, list(MACRO.VALIDATION_RANGE)[0], queue))
                p.start()
                processes.append(p)

            for p in (processes):
                stats.append(queue.get())
                p.join()


            if None in stats: 
                continue

            sumTrueDetections = sum(i[0] for i in stats) 
            sumFalseDetections = sum(i[1] for i in stats)
            sumPredictions = sum(i[2] for i in stats)
            sumGT = sum(i[3] for i in stats)

            try:
                precision = sumTrueDetections / sumPredictions 
            except ZeroDivisionError: 
                precision = 0

            try:
                recall = sumTrueDetections / sumGT
            except ZeroDivisionError: 
                recall = 0


            try: 
                f_score = 2 * (precision*recall) / (precision+recall)
            except: 
                f_score = 0

            print("\nF score: ", f_score)

            if maxResult < f_score: 
                maxResult = f_score
                print("Saving model with F_score %s" % f_score)
                torch.save(model.state_dict(), MACRO.BESTMODELPATH)


  transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])

  if MACRO.VALIDATION_RANGE:  
    validationImagesPaths = [MACRO.GROUNDTRUTHIMAGES + "%s.bmp" % fileIndex for fileIndex in MACRO.VALIDATION_RANGE]
    validationImages = numpy.array([cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in validationImagesPaths])


    model = util.utils.loadModel(
                modelPath = MACRO.BESTMODELPATH, 
                device = MACRO.DEVICE, 
                NUMBERLAYERS = MACRO.NUMBERLAYERS, 
                NUMBERFEATURES = MACRO.NUMBERFEATURES,
                MAXPOOLING = MACRO.MAXPOOLING, 
                WINDOWSIZE=MACRO.WINDOW_SIZE, 
                residual=args.residual, 
                gabriel=args.gabriel, 
                su=args.su 
                )
            # turn on the testing mode
    model.eval()
    model.to(device)

    predictedImages = []  
    for image in validationImages: 
        tensorImage = transforms(image)
        predictionImage = model(tensorImage.float().unsqueeze(dim=0).to(MACRO.DEVICE)).detach().cpu()
        predictedImages.append(predictionImage)

    
    hs = dict()
    prob = MACRO.TEST_START_PROBABILITY


    if MACRO.VALIDATION_RANGE:
        print("\n\n-----------------------------------------")
        print("Testing -- Choosing best probability")

        count = 0
        for prob in tqdm(numpy.arange(MACRO.TEST_START_PROBABILITY, MACRO.TEST_END_PROBABILITY, MACRO.TEST_STEP)):
            if list(MACRO.VALIDATION_RANGE) == [0]: 
                MACRO.BESTMODELPATH = MACRO.MODELPATH
                best_prob = 0.5 
                break
            
            print("---------------------------")
            print("Current probability", prob)  
            entireImage.findPores(
            MACRO.BESTMODELPATH, 
            list(MACRO.VALIDATION_RANGE), 
            prob, 
            MACRO.GROUNDTRUTHIMAGES, 
            MACRO.GROUNDTRUTHLABELS,     
            MACRO.PREDICTIONFINGERPRINT, 
            MACRO.PREDICTIONPORE, 
            MACRO.PREDICTIONCOORDINATES, 
            nsmThreshold=MACRO.NMS_THRESHOLD, 
            nmsWindow=MACRO.WINDOW_SIZE,
            device=torch.device(MACRO.DEVICE), 
            NUMBERLAYERS = MACRO.NUMBERLAYERS, 
            NUMBERFEATURES = MACRO.NUMBERFEATURES,
            MAXPOOLING = MACRO.MAXPOOLING, 
            residual=args.residual, 
            gabriel=args.gabriel, 
            su = args.su,
            boundingBoxSize=args.boundingBoxSize, 
            preDefinedPrediction=copy.deepcopy(predictedImages)
            )

            
            queue = mp.Queue()
            processes = []
            stats = []
            for i in MACRO.VALIDATION_RANGE: 
                p = mp.Process(target=  validate.test, args=(i, MACRO.GROUNDTRUTHLABELS, MACRO.PREDICTIONCOORDINATES, -1, -1, MACRO.WINDOW_SIZE, list(MACRO.VALIDATION_RANGE)[0], queue))
                p.start()
                processes.append(p)

            for p in (processes):
                stats.append(queue.get())
                p.join()


            if None in stats: 
                continue

            sumTrueDetections = sum(i[0] for i in stats) 
            sumFalseDetections = sum(i[1] for i in stats)
            sumPredictions = sum(i[2] for i in stats)
            sumGT = sum(i[3] for i in stats)

            try:
                precision = sumTrueDetections / sumPredictions 
            except ZeroDivisionError: 
                precision = 0

            try:
                recall = sumTrueDetections / sumGT
            except ZeroDivisionError: 
                recall = 0


            try: 
                f_score = 2 * (precision*recall) / (precision+recall)
            except: 
                f_score = 0

            print("\nF score: ", f_score)

            hs[prob] = f_score

            if len(hs) > 2:
                prevRes = max(hs.values())
                if prevRes > f_score: 
                    count += 1 

                if count > MACRO.TOLERANCE: 
                    print("Stopping early")
                    break



        if len(hs) != 0:           
            print(max(hs, key=hs.get), hs[max(hs, key=hs.get)])
            best_prob = max(hs, key=hs.get)

        print("\n\n-----------------------------------------")
        print("Testing -- Choosing best NMS Threshold")
        hs = dict()
        count = 0
        for prob in tqdm(numpy.arange(MACRO.TEST_START_NMS, MACRO.TEST_END_NMS, MACRO.STEP_NMS)):
            if list(MACRO.VALIDATION_RANGE) == [0]: 
                MACRO.BESTMODELPATH = MACRO.MODELPATH
                best_NMS = 0 
                best_prob = 0.5
                break

            print("---------------------------")
            print("Current NMS threshold: ", prob)
            entireImage.findPores(
            MACRO.BESTMODELPATH, 
            MACRO.VALIDATION_RANGE, 
            best_prob,
            MACRO.GROUNDTRUTHIMAGES, 
            MACRO.GROUNDTRUTHLABELS,     
            MACRO.PREDICTIONFINGERPRINT, 
            MACRO.PREDICTIONPORE, 
            MACRO.PREDICTIONCOORDINATES, 
            nsmThreshold=prob, 
            nmsWindow=MACRO.WINDOW_SIZE,
            device=torch.device(MACRO.DEVICE), 
            NUMBERLAYERS = MACRO.NUMBERLAYERS, 
            NUMBERFEATURES = MACRO.NUMBERFEATURES,
            MAXPOOLING = MACRO.MAXPOOLING, 
            residual=args.residual, 
            gabriel=args.gabriel, 
            su = args.su, 
            boundingBoxSize=args.boundingBoxSize

            )

            ans1, ans2 = 0, 0 
            queue = mp.Queue()
            processes = []
            stats = []
            for i in MACRO.VALIDATION_RANGE: 
                p = mp.Process(target=  validate.test, args=(i, MACRO.GROUNDTRUTHLABELS, MACRO.PREDICTIONCOORDINATES, -1, -1, MACRO.WINDOW_SIZE, list(MACRO.VALIDATION_RANGE)[0], queue))
                p.start()
                processes.append(p)

            for p in (processes):
                stats.append(queue.get())
                p.join()


            if None in stats: 
                continue

            sumTrueDetections = sum(i[0] for i in stats) 
            sumFalseDetections = sum(i[1] for i in stats)
            sumPredictions = sum(i[2] for i in stats)
            sumGT = sum(i[3] for i in stats)

            try:
                precision = sumTrueDetections / sumPredictions 
            except ZeroDivisionError: 
                precision = 0

            try:
                recall = sumTrueDetections / sumGT
            except ZeroDivisionError: 
                recall = 0


            try: 
                f_score = 2 * (precision*recall) / (precision+recall)
            except: 
                f_score = 0

            hs[prob] = f_score    

            print("\nF score: ", f_score)

            if len(hs) > 2:
                prevRes = max(hs.values())
                if prevRes > f_score: 
                    count += 1 
                    print(count)

                if count > MACRO.TOLERANCE: 
                    print("\nStopping early")
                    break

        if len(hs) != 0: 
            best_NMS = max(hs, key=hs.get)


  if MACRO.TEST_RANGE:   
    print("=======================================================")
    print("TEST I")

    entireImage.findPores(
    MACRO.BESTMODELPATH if os.path.isfile(MACRO.BESTMODELPATH) else MACRO.MODELPATH, 
    MACRO.TEST_RANGE, 
    best_prob if best_prob != None else args.defaultProb,
    MACRO.GROUNDTRUTHIMAGES, 
    MACRO.GROUNDTRUTHLABELS,     
    MACRO.PREDICTIONFINGERPRINT, 
    MACRO.PREDICTIONPORE, 
    MACRO.PREDICTIONCOORDINATES, 
    nsmThreshold=best_NMS if best_NMS != None else args.defaultNMS, 
    nmsWindow=MACRO.WINDOW_SIZE,
    device=torch.device(MACRO.DEVICE), 
    NUMBERLAYERS = MACRO.NUMBERLAYERS, 
    NUMBERFEATURES = MACRO.NUMBERFEATURES,
    MAXPOOLING = MACRO.MAXPOOLING, 
    residual=args.residual, 
    gabriel=args.gabriel, 
    su = args.su, 
    boundingBoxSize=args.boundingBoxSize
    )



    ans1, ans2 = 0, 0 
    queue = mp.Queue()
    processes = []
    stats = []
    for i in MACRO.TEST_RANGE: 
        p = mp.Process(target=  validate.test, args=(i, MACRO.GROUNDTRUTHLABELS, MACRO.PREDICTIONCOORDINATES, -1, -1, MACRO.WINDOW_SIZE, list(MACRO.TEST_RANGE)[0], queue))
        p.start()
        processes.append(p)

    for p in (processes):
        stats.append(queue.get())
        p.join()


    sumTrueDetections = sum(i[0] for i in stats) 
    sumFalseDetections = sum(i[1] for i in stats)
    sumPredictions = sum(i[2] for i in stats)
    sumGT = sum(i[3] for i in stats)

    try:
        precision = sumTrueDetections / sumPredictions 
    except ZeroDivisionError: 
        precision = 0

    try:
        recall = sumTrueDetections / sumGT
    except ZeroDivisionError: 
        recall = 0


    try: 
        f_score = 2 * (precision*recall) / (precision+recall)
    except: 
        f_score = 0



    hs[int(prob*100)/100] = f_score

    print("F score", str(f_score))
    print("True Detection Rate:", precision)
    print("False Detection Rate:", 1.00-recall)

  if args.secondtestingRange:
    print("---------------------------")
    print("TEST II")

    entireImage.findPores(
    MACRO.BESTMODELPATH if os.path.isfile(MACRO.BESTMODELPATH) else MACRO.MODELPATH, 
    args.secondtestingRange, 
    best_prob if best_prob != None else args.defaultProb,
    MACRO.GROUNDTRUTHIMAGES, 
    MACRO.GROUNDTRUTHLABELS,     
    MACRO.PREDICTIONFINGERPRINT, 
    MACRO.PREDICTIONPORE, 
    MACRO.PREDICTIONCOORDINATES, 
    nsmThreshold=best_NMS if best_NMS != None else args.defaultNMS, 
    nmsWindow=MACRO.WINDOW_SIZE,
    device=torch.device(MACRO.DEVICE), 
    NUMBERLAYERS = MACRO.NUMBERLAYERS, 
    NUMBERFEATURES = MACRO.NUMBERFEATURES,
    MAXPOOLING = MACRO.MAXPOOLING, 
    residual=args.residual, 
    gabriel=args.gabriel, 
    su = args.su, 
    boundingBoxSize=args.boundingBoxSize
    )


    

    ans1, ans2 = 0, 0 
    queue = mp.Queue()
    processes = []
    stats = []
    for i in args.secondtestingRange: 
        p = mp.Process(target=  validate.test, args=(i, MACRO.GROUNDTRUTHLABELS, MACRO.PREDICTIONCOORDINATES, -1, -1, MACRO.WINDOW_SIZE, list(range(121, 151))[0], queue))
        p.start()
        processes.append(p)

    for p in (processes):
        stats.append(queue.get())
        p.join()


    sumTrueDetections = sum(i[0] for i in stats) 
    sumFalseDetections = sum(i[1] for i in stats)
    sumPredictions = sum(i[2] for i in stats)
    sumGT = sum(i[3] for i in stats)

    try:
        precision = sumTrueDetections / sumPredictions 
    except ZeroDivisionError: 
        precision = 0

    try:
        recall = sumTrueDetections / sumGT
    except ZeroDivisionError: 
        recall = 0


    try: 
        f_score = 2 * (precision*recall) / (precision+recall)
    except: 
        f_score = 0



    hs[int(prob*100)/100] = f_score

    print("F score", str(f_score))
    print("True Detection Rate:", precision)
    print("False Detection Rate:", 1.00-recall)



import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    algorithm()





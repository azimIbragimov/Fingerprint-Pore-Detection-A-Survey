import cv2
import numpy 
import random


class EntryGiver: 

    table = {}

    def __init__(self, 
        GROUNDTRUTHIMAGES, 
        GROUNDTRUTHLABELS,
        TRAIN_RANGE, 
        PORELABELRADIUS, 
        WINDOW_SIZE, 
        softLabels
        ):
        
        fileNames = [str(i+1) for i in TRAIN_RANGE]

        imageFilenames = [GROUNDTRUTHIMAGES + str(i+1) + ".bmp" for i in TRAIN_RANGE]
        labelFilenames = [GROUNDTRUTHLABELS + str(i+1) + ".txt" for i in TRAIN_RANGE]

        c = list(zip(imageFilenames, labelFilenames))
        random.shuffle(c)
        imageFilenames, labelFilenames = zip(*c)

        


        currentImage = cv2.imread(imageFilenames[0], cv2.IMREAD_GRAYSCALE)
        labelsTable = [numpy.zeros(currentImage.shape) for i in TRAIN_RANGE]

        numberPores = 0
        for index, file in enumerate(labelFilenames):
            with open(file) as f:
                lines = f.readlines()
                line = None
                for number, i in enumerate(lines):
                    line = i.split()
                    if not line[0].isdigit() or not line[1].isdigit(): continue
                    X, Y = int(line[0])-1,int(line[1])-1
                    labelsTable[index][X, Y] = 1
                    self.makeCircle(labelsTable[index], PORELABELRADIUS, X, Y, softLabels)

        trueLabels = numpy.count_nonzero(numpy.array(labelsTable)[:, WINDOW_SIZE//2:-WINDOW_SIZE//2, WINDOW_SIZE//2:-WINDOW_SIZE//2])
        print(trueLabels, numberPores)

        zeroTable = [[] for i in TRAIN_RANGE]
        oneTable = [[] for i in TRAIN_RANGE]

        counter = 0
        counter2 = 0
        for number in range(len(labelsTable)): 
            currentImage = cv2.imread(imageFilenames[number], cv2.IMREAD_GRAYSCALE)
            for i in range(currentImage.shape[0]-WINDOW_SIZE): 
                for j in range(currentImage.shape[1]-WINDOW_SIZE):

                    if counter == trueLabels-1 and counter2 < trueLabels - 1:
                        break

                    subImage = currentImage[i : i + WINDOW_SIZE, j : j + WINDOW_SIZE] 
                    label = labelsTable[number][i + WINDOW_SIZE//2][j + WINDOW_SIZE//2]
                    if label == 0: 
                        if counter2 < trueLabels - 1:
                            counter2 +=1
                            zeroTable[number].append([subImage, label]) 
                    else: 
                        if counter < trueLabels - 1:
                            oneTable[number].append([subImage, label]) 
                            counter += 1

        print("Size true: " + str(sum(len(i)for i in oneTable)), "Size false: " + str(sum(len(i)for i in zeroTable)))
        self.table = []

        for number, file in enumerate(oneTable):
            random.shuffle(zeroTable[number])
            file += zeroTable[number]

        self.table = oneTable


        print("Total: ", str(sum(len(i)for i in self.table)))

    def getTable(self): 
        return self.table



    def makeCircle(self, table, radious, locationX, locationY, softLabels):
    
        for i in range(locationX-radious, locationX+radious+1):
            for j in range(locationY-radious, locationY+radious+1):
                if i >= 0 and j >= 0 and i < len(table) and j < len(table[0]):
                    if (i-locationX)**2 + (j-locationY)**2 <= radious**2:
                        if softLabels: 
                            try:
                                table[i][j] = 1 - (((i-locationX)**2 + (j-locationY)**2)**0.5)/radious
                            except: 
                                table[i][j] = 1
                        else: 
                            table[i][j] = 1






        


            

            





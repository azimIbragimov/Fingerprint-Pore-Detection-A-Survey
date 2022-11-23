import torch
import entryGiver


class datasetPores(torch.utils.data.Dataset):

    def __init__(
        self, 
        TRAIN_RANGE, 
    
        GROUNDTRUTHIMAGES, 
        GROUNDTRUTHLABELS, 
        PORELABELRADIUS, 
        WINDOW_SIZE,
        softLabels, 

        train = True,
        transform = None,
        ):

        self.transform = transform
        self.occurances = [0, 0]

        temptable = entryGiver.EntryGiver(
            GROUNDTRUTHIMAGES, 
            GROUNDTRUTHLABELS,
            TRAIN_RANGE, 
            PORELABELRADIUS, 
            WINDOW_SIZE,
            softLabels
        )
        self.table = temptable.getTable()

        self.trainTable = [] 
        self.testTable = []

        for file in range(len(TRAIN_RANGE)): 
            for entry in self.table[file]: 
                self.trainTable.append(entry)


        self.table = self.trainTable 



    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        
        image = self.table[index][0]
        label = self.table[index][1]

        self.occurances[int(label)] += 1

        if(self.transform): 
            image = self.transform(image)

        return (image, label)

    def returnOccurances(self): 
        return self.occurances
    
        
    def get_labels(self):
        labelTable = []

        for entry in self.table:

            image, label = entry[0], entry[1]
            labelTable.append(label)

        return labelTable
            


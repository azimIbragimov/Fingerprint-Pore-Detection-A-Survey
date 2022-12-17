import os, argparse, shutil, re
from fnmatch import fnmatch
from tqdm import tqdm

class Dataset(): 

    def __init__(self, initialDir, finalDir, substractOne):
        self.storage = []
        self.initialDir = initialDir
        self.finalDir = finalDir
        self.substractOne = substractOne

    def append(self, image, label):
        self.storage.append((image, label)) 

    def __str__(self) -> str:
        for i in self.storage: 
            print(i)

        return " " 

    def write(self): 
        
        os.system(f"mkdir -p {self.finalDir}/PoreGroundTruthSampleimage")
        os.system(f"mkdir -p {self.finalDir}/PoreGroundTruthMarked")

        for i, file in tqdm(enumerate(self.storage)):
            shutil.copy(file[0], self.finalDir + f"/PoreGroundTruthSampleimage/{i+1}.bmp")
            shutil.copy(file[1], self.finalDir + f"/PoreGroundTruthMarked/{i+1}.txt")

            with open(self.finalDir + f"/PoreGroundTruthMarked/{i+1}.txt", "r") as f:
                lines = f.readlines()

                for l in range(len(lines)): 
                    X, Y = lines[l].split()
                    if not X.isdigit() or not Y.isdigit(): continue

                    if not self.substractOne:
                        lines[l] = f"{Y}\t{X}\n"
                    else: 
                        lines[l] = f"{int(X)-1}\t{int(Y)-1}\n"

                f.close()

            with open(self.finalDir + f"/PoreGroundTruthMarked/{i+1}.txt", "w") as f:
                f.truncate(0)
                f.writelines(lines)
                f.close()




def numeric_sort_key(s):
    return int(re.match(r'\d+', s).group())


                

if __name__=="__main__": 

    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetDirectory', 
                    required=True,
                    type=str, 
                    )  

    parser.add_argument('--imageExtension', 
                    required=True,
                    type=str, 
                    )  

    parser.add_argument('--labelsExtension', 
                    required=True,
                    type=str, 
                    )  
    parser.add_argument('--transformDirectory', 
                        required=True,
                        type=str,
                        )

    parser.add_argument('--substractOne', 
                        default=False, 
                        action='store_true',
                        )

    

    args = parser.parse_args()

    patternLabel = f'*.{args.labelsExtension}'
    patternImage = f'*.{args.imageExtension}'


    imageList, labelsList = [], []


    # getting labels
    for path, subdirs, files in os.walk(args.datasetDirectory):
        for name in sorted(files):
            if fnmatch(name, patternLabel):
                labelsList.append(os.path.join(path, name))


    for path, subdirs, files in os.walk(args.datasetDirectory):
        for name in sorted(files):
            if fnmatch(name, patternImage):
                imageList.append(os.path.join(path, name))

    
    dataset = Dataset(args.datasetDirectory, args.transformDirectory, args.substractOne)


    for i in (range(len(labelsList))): 
        dataset.append(imageList[i], labelsList[i])

    dataset.write()


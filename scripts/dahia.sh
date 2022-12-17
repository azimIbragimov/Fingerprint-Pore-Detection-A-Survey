chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/dahia 

# :'
# 1. input image has dimensions 17 by 17

# 2. The probability of an image patch Ip of dimensions 17 by 17 having a pore is 1 if 
# there is a pore within a bounding box of di- mensions 7 by 7 centered in Ips center;  

# 3. Training, then, is formulated as minimizing the cross-entropy between the models 
# predictions and the training data pore probability distribution; Thus the loss function 
# is cross entrapy 

# 4. Patch sampling consists of sampling patches randomly from all images of the 
# training set to perform gradient-based learn- ing, instead of using entire images 
# in mini-batches; Thus random sampler was used 

# 5. We then construct the set B of 7 by 7 bounding boxes around each spatial coordinate (i, j) 
# for which P (i, j) is above a pre-determined threshold pt.The procedure outlined so far connects 
# the bounding box detections of the same pore by the overlap in their areas. Hence, we apply NMS 
# to convert them into a single detec- tion; Therefore NMS with bounding boxes of 7 by 7 were used

# 6. Its 30 images should be split with the first 15 images forming the training set, the next 5 
# the validation set, and the last 10 the test set;

# 7. Bidirectional correspondence 

# 8. All the neural network models are optimized using Stochastic Gradient Descent (SGD) with early stopping.
# 9. We manually tune the learning rate and its decay sched- ule, the batch size, the dropout rate and the 
# amount of weight decay by measuring the F-score of the patches clas- sified as pores using the training labels. 
# We set their values at 10^-1, exponentially decayed by 0.96 every 2000 steps, 256, 0.2 and 0, respectively
# 10. The post-processing parame- ters are obtained by performing a grid search, optimizing validation F-score 
# with the trained model for pt and the NMS intersection threshold it. The range of the search for pt is 
# {0.1, 0.2, ..., 0.9} and for it is {0, 0.1, ..., 0.7}. The chosen values were pt = 0.6 and it = 0. In our 
# implemen- tation of NMS, it = 0 corresponds to discarding bounding boxes that have any amount of intersection.
# '

python3 train.py --patchSize 17 --boundingBoxSize 7 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-15 --validationRange 16-20 --testingRange 21-30  --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.1 --testEndProbability 0.9 --testStepProbability 0.1 --testStartNMSUnion 0.0 --testStepNMSUnion 0.1  --testEndNMSUnion 0.9  --experimentPath experiments/dahia  --gabriel --device cuda --numberWorkers 8 
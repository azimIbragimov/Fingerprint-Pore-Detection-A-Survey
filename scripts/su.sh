chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/su 

# 1. The first 20 images were used for training while the remaining for testing. for training,
#2. If a image patch is centered within 3 pixels from the given ground truth of pore positions' it is labeled as 1; otherwise 0.
#3. The experiment in this sub-section was performed based on N=17

python3 train.py --patchSize 17 --boundingBoxSize 17 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-15 --validationRange 16-20 --testingRange 21-30  --criteriation BCELOSS --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.0 --testEndProbability 1.0  --testStartNMSUnion 0.0   --experimentPath experiments/su  --su --device cuda >> experiments/su/exp.log
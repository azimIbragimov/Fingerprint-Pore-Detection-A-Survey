chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/softLabels 

python3 train.py --patchSize 17 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3  --experimentPath experiments/softLabels --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda  --softLabels  >> experiments/softLabels/softexp1.log; wait;
python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3  --experimentPath experiments/softLabels --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda  --softLabels >> experiments/softLabels/softexp2.log; wait;

python3 train.py --patchSize 17 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3  --experimentPath experiments/softLabels --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda   --numberWorkers 8 >> experiments/softLabels/nosoftexp3.log; wait;
python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3  --experimentPath experiments/softLabels --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda   --numberWorkers 8 >> experiments/softLabels/nosoftexp4.log; wait;
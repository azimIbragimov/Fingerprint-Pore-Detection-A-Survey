
chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/baseline 

python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3  --experimentPath experiments/baseline --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels --numberWorkers 2 --numberFeatures 40 --epoc 2 >> experiments/baseline/exp.log; wait;

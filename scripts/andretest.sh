chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/l3sf 



python3 train.py --patchSize 15 --poreRadius 5 --groundTruth dataset --trainingRange 1-444 --validationRange 445-592 --testingRange 593-740 --experimentPath experiments/first --testStartProbability 0.3 --testStartNMSUnion 0.15 --boundingBoxSize 13 --device cuda  --softLabels > experiments/l3sf/l3sf1.log; wait;
python3 train.py --patchSize 15 --poreRadius 5 --groundTruth dataset --trainingRange 149-592 --validationRange 593-740 --testingRange 1-148 --experimentPath experiments/second --testStartProbability 0.3 --testStartNMSUnion 0.15 --boundingBoxSize 13 --device cuda  --softLabels > experiments/l3sf/l3sf2.log; wait;
python3 train.py --patchSize 15 --poreRadius 5 --groundTruth dataset --trainingRange 297-739 --validationRange 1-148 --testingRange 149-296 --experimentPath experiments/third --testStartProbability 0.3 --testStartNMSUnion 0.15 --boundingBoxSize 13 --device cuda:2  --softLabels > experiments/l3sf/l3sf3.log;; wait;
python3 train.py --patchSize 15 --poreRadius 5 --groundTruth dataset --trainingRange 445-739,1-148 --validationRange 149-296 --testingRange 297-444 --experimentPath experiments/forth --testStartProbability 0.3 --testStartNMSUnion 0.15 --boundingBoxSize 13 --device cuda  --softLabels > experiments/l3sf/l3sf4.log;; wait;
python3 train.py --patchSize 15 --poreRadius 5 --groundTruth dataset --trainingRange 593-739,1-296 --validationRange 297-444 --testingRange 445-592 --experimentPath experiments/fifth --testStartProbability 0.3 --testStartNMSUnion 0.15 --boundingBoxSize 13 --device cuda  --softLabels > experiments/l3sf/l3sf5.log;; wait;

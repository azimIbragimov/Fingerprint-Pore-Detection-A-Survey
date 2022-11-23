chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/15PatchSize 

python3 train.py --patchSize 15 --poreRadius 3 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 15 --device cuda  --softLabels    >> experiments/15PatchSize/L3SFexp17.log;
python3 train.py --patchSize 15 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 15 --device cuda  --softLabels    >> experiments/15PatchSize/L3SFexp18.log; wait;
python3 train.py --patchSize 15 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 15 --device cuda  --softLabels    >> experiments/15PatchSize/L3SFexp19.log; wait;
python3 train.py --patchSize 15 --poreRadius 6 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp20.log; wait;

python3 train.py --patchSize 15 --poreRadius 3 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp21.log; wait;
python3 train.py --patchSize 15 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp22.log; wait;
python3 train.py --patchSize 15 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp23.log; wait;
python3 train.py --patchSize 15 --poreRadius 6 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp24.log; wait;

python3 train.py --patchSize 15 --poreRadius 3 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp25.log; wait;
python3 train.py --patchSize 15 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp26.log; wait;
python3 train.py --patchSize 15 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp27.log; wait;
python3 train.py --patchSize 15 --poreRadius 6 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp28.log; wait;

python3 train.py --patchSize 15 --poreRadius 3 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp29.log; wait;
python3 train.py --patchSize 15 --poreRadius 4 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp30.log; wait;
python3 train.py --patchSize 15 --poreRadius 5 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp31.log; wait;
python3 train.py --patchSize 15 --poreRadius 6 --groundTruth dataset --trainingRange $1 --validationRange $2 --testingRange $3   --maxPooling True --experimentPath experiments/15PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 15 --device cuda --softLabels    >> experiments/15PatchSize/L3SFexp32.log; wait;
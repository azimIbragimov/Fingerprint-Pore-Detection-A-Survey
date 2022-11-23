chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/dpf 

python3 dpf.py  --groundTruth dataset --testingRange 1-30  --experimentPath experiments/dpf
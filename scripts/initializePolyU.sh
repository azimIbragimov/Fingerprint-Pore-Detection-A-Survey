chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/dataset

python3 scripts/initializeDataset.py --datasetDirectory $1  --imageExtension bmp --labelsExtension txt --transformDirectory dataset --substractOne
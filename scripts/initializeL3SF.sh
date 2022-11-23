chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/dataset

python3 scripts/initializeDataset.py --datasetDirectory $1/Pore\ ground\ truth --imageExtension jpg --labelsExtension tsv --transformDirectory dataset
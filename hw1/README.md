# Usage
## First generate training data from expert policy
```
cd hw1/
python3 run_expert.py experts/Hopper-v1.pkl Hopper-v1
```
This will generate the training data/label file called "Hopper-v1" in "experts_data" folder, which we used for training phase in BC and DAgger strategies.

## For Behaviour Cloning (BC)
```
python3 BC.py experts/Hopper-v1.pkl Hopper-v1 --render
```

## For Data Aggregation (DAgger)
```
python3 DAgger.py experts/Hopper-v1.pkl Hopper-v1 --render
```
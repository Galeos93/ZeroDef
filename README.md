```
make download-data
```

```
make create-extended-dataset
```

```
make split-dataset DATASET=/home/agarcia/repos/ZeroDef/zero_deforestation/temp/extended_data.csv TRAIN_PROPORTION=0.9
PYTHONPATH=. python zero_deforestation/scripts/split_train_val.py /home/agarcia/repos/ZeroDef/zero_deforestation/temp/extended_data.csv 0.9
```

```
PYTHONPATH=. python zero_deforestation/train.py --c zero_deforestation/final_solution_config.json
```
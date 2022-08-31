# Classifier Beard-Mustache-Eyeglasses

## Preparing datasets

## Training phase

Before training, make sure that the 
following value are set in 'config.yml':

- features -> with/without
- libs -> predictor
- dataset -> training
- models

To start the training:

```
python train_model.py
```

After training, the resulting models 
can be used for the test phase are saved in:

```.
|-- models
|   |-- beard
|   |   |-- predictor.pkl
|   |-- glasses
|   |   |-- predictor.pkl
|   |-- mustache
|   |   |-- predictor.pkl
```

## Test phase

Before test, make sure that the 
following value are set in 'config.yml':

- features -> with/without
- libs -> predictor
- dataset -> test/folder
- labels -> test/results
- output -> results
- models

To start the test:

```
python test.py
```

To start the test with 
the following custom parameter (using cli):

- data -> Full path to Dataset labels
- images -> Path to Dataset folder
- results -> Name of CSV file of the results

Example:
```
python test.py --data ./foo/foo.csv --images ./foo/ --results results.csv
```

After test, the result can be used for 
the evaluation phase are saved in:

```.
|-- output
|   |-- results.csv
```

## Evaluation metrics

Before Evaluation, make sure that the 
following value are set in 'config.yml':

- dataset -> folder
- labels -> test/results
- output -> results

To start the evaluation:

```
python evaluate.py
```

To start the evaluation with 
the following custom parameter (using cli):

- gt_path -> Fullpath to File CSV with groundtruth
- res_path -> Fullpath to File CSV with prediction results

Example:
```
python test.py --gt_path ./foo/foo.csv --res_path ./foo2/foo2.csv
```

## License

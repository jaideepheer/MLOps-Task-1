# MLOps-Task-1

## def test_model_writing()

> TODO: write  a test case to check if model is successfully getting created or not?

    1. create some data

    2. run_classification_experiment(data, expeted-model-file)

    assert os.path.isfile(expected-model-file)

## def test_small_data_overfit_checking()

> TODO: write a test case to check fitting on training -- litmus test.

    1. create a small amount of data / (digits / subsampling)

    2. train_metrics = run_classification_experiment(train=train, valid=train)

    assert train_metrics['acc']  > some threshold

    assert train_metrics['f1'] > some other threshold

## Test Output

```
(/mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1/.venv) jaideep@JD-GPC:/mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1$ pytest tests
==================================================================
test session starts
===================================================================
platform linux -- Python 3.8.5, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
rootdir: /mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1
collected 8 items

tests/test_model_writing.py ... [ 37%]
tests/test_samples.py .....     [100%]

===================================================================
8 passed in 6.00s
====================================================================
```

## Run Output

```
(/mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1/.venv) jaideep@JD-GPC:/mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1$ python -m mlops_task_1.plot_graphs
Original image size: (1797, 8, 8)
Models folder (models) exists.
Do you want to overwrite data?[Y/n]:y
Skipping for gamma=1e-07.
Skipping for gamma=1e-06.
Skipping for gamma=1e-05.
2x2     0.001   85.0    15.0    0.7142857142857143      0.566137566137566
Skipping for gamma=1e-07.
Skipping for gamma=1e-06.
Skipping for gamma=1e-05.
4x4     0.0001  85.0    15.0    0.9047619047619048      0.825
Skipping for gamma=1e-07.
Skipping for gamma=1e-06.
Skipping for gamma=0.1.
8x8     0.001   85.0    15.0    1.0     1.0
Skipping for gamma=1e-07.
Skipping for gamma=0.1.
16x16   0.0001  85.0    15.0    1.0     1.0
Skipping for gamma=1e-07.
Skipping for gamma=0.01.
Skipping for gamma=0.1.
24x24   0.0001  85.0    15.0    1.0     1.0
Skipping for gamma=1e-07.
Skipping for gamma=1e-06.
Skipping for gamma=1e-05.
2x2     0.001   80.0    20.0    0.75    0.5952380952380951
Skipping for gamma=1e-07.
Skipping for gamma=1e-06.
Skipping for gamma=1e-05.
4x4     0.0001  80.0    20.0    0.9166666666666666      0.8285714285714285
Skipping for gamma=1e-07.
Skipping for gamma=1e-06.
Skipping for gamma=0.1.
8x8     0.001   80.0    20.0    1.0     1.0
Skipping for gamma=1e-07.
Skipping for gamma=0.1.
16x16   0.0001  80.0    20.0    1.0     1.0
Skipping for gamma=1e-07.
Skipping for gamma=0.01.
Skipping for gamma=0.1.
24x24   0.0001  80.0    20.0    1.0     1.0
```
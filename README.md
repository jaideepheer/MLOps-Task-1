# MLOps-Task-1

Write a test cases for the `create_split` function. check for number of samples in each split, and sum of samples in each split should come to the total number of samples. (so, 4 assert statements)

1. if you give n=100 samples, and provide train:test:valid split as 70:20:10, there should be 70,20, and 10 samples in the train, tests, and validation splits returned by `create_split` function.

2. given n=9 samples, with 70:20:10 split, the train should contain 6, 2, 1 samples respectively.

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
8 passed in 5.99s
====================================================================
```

# MLOps-Task-1

Hi all,
In today's lab session, we will do experiments to understand "how to identify if model A is better than model B or not?"

- We will add a functionality to use "DecisionTree" classifier https://scikit-learn.org/stable/modules/tree.html#classification
- We will have 5 different splits of train/test/valid, and will report performance of existing SVM and the new DecisionTree classifier. (question: what about hyper-parameter turning).
- We will compute the mean and standard deviations of bot the classifier's performances.
- Interpret the numbers
- Identify if there is more to the classifier comparison than just the numbers?

## How to run

```
python -m mlops_task_1.plot_compare
```

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

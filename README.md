# MLOps-Task-1

In today's class we will continue on the themes of "debugging and cleaning" and "model selection"

- we will modify the code such that for a hyper-parameter configuration, the model is trained not more than once.
- we will ensure to throw away some of the models that yield random-like performance.
- we will save and retrieve models from hard disk, instead of keeping everything in memory
- we will modularize the code to have following functions: preprocess, train, create_splits, validate, test, report
- bonus: can you remove all the hardcoded values from the py file, and make them as arguments?

## Output

```
Original image size: (1797, 8, 8)
Models folder (models) exists.
Do you want to overwrite data?[Y/n]:
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
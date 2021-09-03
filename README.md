# MLOps-Task-1

For today's class we will focus on "data processing" hands on.

In the same/similar example of scikit-learn,

1. check what is the "image" size in the digits dataset
2. resize images to at least 3 different resolutions : take help from https://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
3. check various train/test splits
4. report how the metrics changes as the image resolution and train/test split changes

## Quiz 1

Code available in [`/mlops_task_1/quiz_1.py`](/mlops_task_1/quiz_1.py)

## Output

```
Original image size: (1797, 8, 8)
Image_Size --> train/test --> Accuracy
(64, 64) --> 0.9/0.1 --> 0.12777777777777777
(64, 64) --> 0.8/0.2 --> 0.1388888888888889
(64, 64) --> 0.7/0.3 --> 0.11296296296296296
(64, 64) --> 0.6/0.4 --> 0.11265646731571627
(32, 32) --> 0.9/0.1 --> 0.7833333333333333
(32, 32) --> 0.8/0.2 --> 0.775
(32, 32) --> 0.7/0.3 --> 0.7574074074074074
(32, 32) --> 0.6/0.4 --> 0.7552155771905424
(16, 16) --> 0.9/0.1 --> 0.9666666666666667
(16, 16) --> 0.8/0.2 --> 0.9527777777777777
(16, 16) --> 0.7/0.3 --> 0.9648148148148148
(16, 16) --> 0.6/0.4 --> 0.9624478442280946
```

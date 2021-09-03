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
Image_Size --> train/test --> Accuracy --> F1 Score (micro)
(64, 64) --> 0.9/0.1 --> 0.128 --> 0.128
(64, 64) --> 0.8/0.2 --> 0.139 --> 0.139
(64, 64) --> 0.7/0.3 --> 0.113 --> 0.113
(64, 64) --> 0.6/0.4 --> 0.113 --> 0.113
(32, 32) --> 0.9/0.1 --> 0.783 --> 0.783
(32, 32) --> 0.8/0.2 --> 0.775 --> 0.775
(32, 32) --> 0.7/0.3 --> 0.757 --> 0.757
(32, 32) --> 0.6/0.4 --> 0.755 --> 0.755
(16, 16) --> 0.9/0.1 --> 0.967 --> 0.967
(16, 16) --> 0.8/0.2 --> 0.953 --> 0.953
(16, 16) --> 0.7/0.3 --> 0.965 --> 0.965
(16, 16) --> 0.6/0.4 --> 0.962 --> 0.962
```

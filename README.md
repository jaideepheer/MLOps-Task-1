# MLOps-Task-1

Hi all,
In today's class we will have a 1hr quiz. The quiz question is :

Note: load the best model of SVM and Decision Tree  from the disc -- do not train them during the test case. For both these models.
1. Add one positive test case per class. For example, "def test_digit_correct_0" function tests if the prediction of an actual digit-0 sample indeed 0 or not, i.e. `assert prediction==0`. (Total of 10 such test cases per model)
2. [Bonus] Add a test case that checks that accuracy on each class is greater than a certain threshold. i.e. `assert acc_digit[0] > min_acc_req`

## How to run

```
pytest tests
```

## Output

```
(/mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1/.venv) jaideep@JD-GPC:/mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1$ pytest tests
================================================================== test session starts ===================================================================
platform linux -- Python 3.8.5, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
rootdir: /mnt/c/Users/jaide/Desktop/New_folder/MLOps-Task-1
collected 18 items                                                                                                                                       

tests/test_best_models.py .FFFFF.FFF                                                                                                               [ 55%]
tests/test_model_writing.py ...                                                                                                                    [ 72%]
tests/test_samples.py .....                                                                                                                        [100%]

======================================================================== FAILURES ========================================================================
______________________________________________________________________ test_digit_1 ______________________________________________________________________

    def test_digit_1():
        DIGIT = 1
        im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
        for m in models:
            mdl = load(root / f'{m}.gz')
>           assert mdl['model'].predict(im) == DIGIT
E           assert array([6]) == 1
E            +  where array([6]) = <bound method BaseSVC.predict of SVC(gamma=0.1)>(array([[ 0.,  0.,  0., 12., 13.,  5.,  0.,  0.,  0.,  0.,  0., 11., 16.,\n         9.,  0.,  0.,  0.,  0.,  3., 15., 16... 1., 16., 16.,  6.,  0.,  0.,  0.,  0.,  1., 16.,\n        16.,  6.,  0.,  0.,  0.,  0.,  0., 11., 16., 10.,  0.,  0.]]))
E            +    where <bound method BaseSVC.predict of SVC(gamma=0.1)> = SVC(gamma=0.1).predict

tests/test_best_models.py:24: AssertionError
______________________________________________________________________ test_digit_2 ______________________________________________________________________

    def test_digit_2():
        DIGIT = 2
        im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
        for m in models:
            mdl = load(root / f'{m}.gz')
>           assert mdl['model'].predict(im) == DIGIT
E           assert array([6]) == 2
E            +  where array([6]) = <bound method BaseSVC.predict of SVC(gamma=0.1)>(array([[ 0.,  0.,  0.,  4., 15., 12.,  0.,  0.,  0.,  0.,  3., 16., 15.,\n        14.,  0.,  0.,  0.,  0.,  8., 13.,  8...16., 16.,  5.,  0.,  0.,  0.,  0.,  3., 13., 16.,\n        16., 11.,  5.,  0.,  0.,  0.,  0.,  3., 11., 16.,  9.,  0.]]))
E            +    where <bound method BaseSVC.predict of SVC(gamma=0.1)> = SVC(gamma=0.1).predict

tests/test_best_models.py:32: AssertionError
______________________________________________________________________ test_digit_3 ______________________________________________________________________

    def test_digit_3():
        DIGIT = 3
        im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
        for m in models:
            mdl = load(root / f'{m}.gz')
>           assert mdl['model'].predict(im) == DIGIT
E           assert array([6]) == 3
E            +  where array([6]) = <bound method BaseSVC.predict of SVC(gamma=0.1)>(array([[ 0.,  0.,  7., 15., 13.,  1.,  0.,  0.,  0.,  8., 13.,  6., 15.,\n         4.,  0.,  0.,  0.,  2.,  1., 13., 13... 0.,  0.,  1., 10.,  8.,  0.,  0.,  0.,  8.,  4.,\n         5., 14.,  9.,  0.,  0.,  0.,  7., 13., 13.,  9.,  0.,  0.]]))
E            +    where <bound method BaseSVC.predict of SVC(gamma=0.1)> = SVC(gamma=0.1).predict

tests/test_best_models.py:40: AssertionError
______________________________________________________________________ test_digit_4 ______________________________________________________________________

    def test_digit_4():
        DIGIT = 4
        im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
        for m in models:
            mdl = load(root / f'{m}.gz')
>           assert mdl['model'].predict(im) == DIGIT
E           assert array([6]) == 4
E            +  where array([6]) = <bound method BaseSVC.predict of SVC(gamma=0.1)>(array([[ 0.,  0.,  0.,  1., 11.,  0.,  0.,  0.,  0.,  0.,  0.,  7.,  8.,\n         0.,  0.,  0.,  0.,  0.,  1., 13.,  6...15., 16., 13., 16.,  1.,  0.,  0.,  0.,  0.,  3.,\n        15., 10.,  0.,  0.,  0.,  0.,  0.,  2., 16.,  4.,  0.,  0.]]))
E            +    where <bound method BaseSVC.predict of SVC(gamma=0.1)> = SVC(gamma=0.1).predict

tests/test_best_models.py:48: AssertionError
______________________________________________________________________ test_digit_5 ______________________________________________________________________

    def test_digit_5():
        DIGIT = 5
        im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
        for m in models:
            mdl = load(root / f'{m}.gz')
>           assert mdl['model'].predict(im) == DIGIT
E           assert array([6]) == 5
E            +  where array([6]) = <bound method BaseSVC.predict of SVC(gamma=0.1)>(array([[ 0.,  0., 12., 10.,  0.,  0.,  0.,  0.,  0.,  0., 14., 16., 16.,\n        14.,  0.,  0.,  0.,  0., 13., 16., 15... 0.,  0.,  4., 16.,  9.,  0.,  0.,  0.,  5.,  4.,\n        12., 16.,  4.,  0.,  0.,  0.,  9., 16., 16., 10.,  0.,  0.]]))
E            +    where <bound method BaseSVC.predict of SVC(gamma=0.1)> = SVC(gamma=0.1).predict

tests/test_best_models.py:56: AssertionError
______________________________________________________________________ test_digit_7 ______________________________________________________________________

    def test_digit_7():
        DIGIT = 7
        im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
        for m in models:
            mdl = load(root / f'{m}.gz')
>           assert mdl['model'].predict(im) == DIGIT
E           assert array([6]) == 7
E            +  where array([6]) = <bound method BaseSVC.predict of SVC(gamma=0.1)>(array([[ 0.,  0.,  7.,  8., 13., 16., 15.,  1.,  0.,  0.,  7.,  7.,  4.,\n        11., 12.,  0.,  0.,  0.,  0.,  0.,  8... 0., 16.,  5.,  0.,  0.,  0.,  0.,  0.,  9., 15.,\n         1.,  0.,  0.,  0.,  0.,  0., 13.,  5.,  0.,  0.,  0.,  0.]]))
E            +    where <bound method BaseSVC.predict of SVC(gamma=0.1)> = SVC(gamma=0.1).predict

tests/test_best_models.py:72: AssertionError
______________________________________________________________________ test_digit_8 ______________________________________________________________________

    def test_digit_8():
        DIGIT = 8
        im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
        for m in models:
            mdl = load(root / f'{m}.gz')
>           assert mdl['model'].predict(im) == DIGIT
E           assert array([6]) == 8
E            +  where array([6]) = <bound method BaseSVC.predict of SVC(gamma=0.1)>(array([[ 0.,  0.,  9., 14.,  8.,  1.,  0.,  0.,  0.,  0., 12., 14., 14.,\n        12.,  0.,  0.,  0.,  0.,  9., 10.,  0...16.,  8., 10., 13.,  2.,  0.,  0.,  1., 15.,  1.,\n         3., 16.,  8.,  0.,  0.,  0., 11., 16., 15., 11.,  1.,  0.]]))
E            +    where <bound method BaseSVC.predict of SVC(gamma=0.1)> = SVC(gamma=0.1).predict

tests/test_best_models.py:80: AssertionError
______________________________________________________________________ test_digit_9 ______________________________________________________________________

    def test_digit_9():
        DIGIT = 9
        im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
        for m in models:
            mdl = load(root / f'{m}.gz')
>           assert mdl['model'].predict(im) == DIGIT
E           assert array([7]) == 9
E            +  where array([7]) = <bound method BaseDecisionTree.predict of DecisionTreeClassifier(max_depth=8)>(array([[ 0.,  0., 11., 12.,  0.,  0.,  0.,  0.,  0.,  2., 16., 16., 16.,\n        13.,  0.,  0.,  0.,  3., 16., 12., 10... 0.,  3.,  0.,  9., 11.,  0.,  0.,  0.,  0.,  0.,\n         9., 15.,  4.,  0.,  0.,  0.,  9., 12., 13.,  3.,  0.,  0.]]))
E            +    where <bound method BaseDecisionTree.predict of DecisionTreeClassifier(max_depth=8)> = DecisionTreeClassifier(max_depth=8).predict

tests/test_best_models.py:88: AssertionError
================================================================ short test summary info =================================================================
FAILED tests/test_best_models.py::test_digit_1 - assert array([6]) == 1
FAILED tests/test_best_models.py::test_digit_2 - assert array([6]) == 2
FAILED tests/test_best_models.py::test_digit_3 - assert array([6]) == 3
FAILED tests/test_best_models.py::test_digit_4 - assert array([6]) == 4
FAILED tests/test_best_models.py::test_digit_5 - assert array([6]) == 5
FAILED tests/test_best_models.py::test_digit_7 - assert array([6]) == 7
FAILED tests/test_best_models.py::test_digit_8 - assert array([6]) == 8
FAILED tests/test_best_models.py::test_digit_9 - assert array([7]) == 9
============================================================= 8 failed, 10 passed in 11.59s ==============================================================

```
from sklearn import datasets
from joblib import load
import pathlib

ds = datasets.load_digits()
models = ['SVC', 'DecisionTreeClassifier']
sz = 'x'.join(map(str, list(ds.images.shape[1:])))
root = pathlib.Path(f'./models/best_{sz}')


def test_digit_0():
    DIGIT = 0
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT


def test_digit_1():
    DIGIT = 1
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT


def test_digit_2():
    DIGIT = 2
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT


def test_digit_3():
    DIGIT = 3
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT


def test_digit_4():
    DIGIT = 4
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT


def test_digit_5():
    DIGIT = 5
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT


def test_digit_6():
    DIGIT = 6
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT


def test_digit_7():
    DIGIT = 7
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT


def test_digit_8():
    DIGIT = 8
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT


def test_digit_9():
    DIGIT = 9
    im = ds.data[ds.target.tolist().index(DIGIT)].reshape(1, -1)
    for m in models:
        mdl = load(root / f'{m}.gz')
        assert mdl['model'].predict(im) == DIGIT

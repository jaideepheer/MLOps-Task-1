from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
from addict import Dict as D
from itertools import product
from typing import List, Dict
import pathlib
import os


def preprocess_images(images, rescale_factor, anti_aliasing=False):
    return np.array(
        [rescale(x, rescale_factor, anti_aliasing=anti_aliasing)
         for x in images]
    )


def create_ttv_splits(data, target, test_size, valid_size):
    X_train, X_test, Y_train, Y_test = train_test_split(
        data,
        target,
        test_size=round((test_size + valid_size) * len(data)),
        shuffle=True,
        # stratify=target,
    )
    X_test, X_valid, Y_test, Y_valid = train_test_split(
        X_test,
        Y_test,
        test_size=round(valid_size * len(data)),
        shuffle=True,
        # stratify=Y_test,
    )
    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid


def predict_metrics(model, X, Y):
    predicted = model.predict(X)
    acc = metrics.accuracy_score(Y, predicted)
    f1 = metrics.f1_score(Y, predicted, average="macro")
    return D({"acc": acc, "f1": f1, })


def flatten_dict(d: dict, *, prefix=[]):
    """
    Recursively flatten dict with each item being a list having last item as value and remaining items as keys.
    """
    if not isinstance(d, dict):
        return prefix + [d]
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten_dict(v, prefix=prefix + [k])
        elif isinstance(v, (list, tuple,)):
            for idx, v in enumerate(v):
                yield from flatten_dict(v, prefix=prefix + [str(idx)])
        else:
            yield prefix + [k, v]


def serialize_dict_to_filename(
    d: dict, path_seperator="-", key_value_seperator="=", item_seperator="|"
):
    """
    This converts a Dict recursively into a valid filename.
    """
    wk = flatten_dict(d)
    # serialise each path + value pair
    wk = map(
        lambda it: f"{path_seperator.join(it[:-1])}{key_value_seperator}{str(it[-1])}",
        wk,
    )
    return item_seperator.join(wk)


def param_grid_iterator(params: dict):
    """
    Accepts a dict where keys are param names and values are lists of possible values for param.
    Returns an iterator over conbinations of param values.
    Ref: https://stackoverflow.com/a/65392983/10027894
    """
    for vcomb in product(*params.values()):
        yield dict(zip(params.keys(), vcomb))


def hparam_search_model(
    X_train,
    X_valid,
    Y_train,
    Y_valid,
    model_builder,
    model_hparams: Dict[str, List],
    **extra_model_args,
):
    """
    Performs hyper-paramater search on models created by calling
        model_builder(**selected_model_hparams + **extra_model_args)

    model_builder should be a callable which when called with args return a trainable model.
    model_hparams should be a dict with key being hparam arg name and value being a list of all hparam value to search in.

    Note: This is a generator function, thus training is deffered untill actually iterated.
    """
    for params in param_grid_iterator(model_hparams):
        model_dict = train_model(
            X_train,
            X_valid,
            Y_train,
            Y_valid,
            model_builder=model_builder,
            **params,
            **extra_model_args,
        )
        yield params, model_dict


def train_model(X_train, X_valid, Y_train, Y_valid, model_builder=SVC, **model_args):
    """
    model_builder should be a callable which when called with args return a trainable model.
    """
    clf = model_builder(**model_args)
    # Learn on the train subset
    clf.fit(X_train, Y_train)
    # test model
    valid_metrics = predict_metrics(clf, X_valid, Y_valid)
    # return model
    return D({"valid_metrics": valid_metrics, "model": clf, "model_args": model_args, "model_builder": model_builder, })

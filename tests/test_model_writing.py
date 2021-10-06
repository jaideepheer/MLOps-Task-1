import pathlib
import mlops_task_1.utils as utils
import mlops_task_1.plot_graphs as plot
from sklearn import datasets
import pathlib

# Returns default test splits for: X_train, X_test, X_valid, Y_train, Y_test, Y_valid
def test_splits(test_size=0.2, valid_size=0.2):
    # get data
    digits = datasets.load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    # create splits
    X_train, X_test, X_valid, Y_train, Y_test, Y_valid = utils.create_ttv_splits(
        data, digits.target, test_size, valid_size
    )
    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid


# Test to check if model file is created or not.
def test_model_writing(tmp_path: pathlib.Path):
    # params
    scaling = 0
    test_size = valid_size = 0.2
    # create splits
    X_train, X_test, X_valid, Y_train, Y_test, Y_valid = test_splits(test_size, valid_size)
    # create model
    model = plot.train_model(X_train, X_valid, Y_train, Y_valid, gamma=0.5)
    model = {
        **model,
        "scaling": scaling,
        "test_size": test_size,
        "valid_size": valid_size,
    }
    # write model
    plot.write_trained_model(model, tmp_path, scaling)
    # check for specific pattern of model files
    out_file = (
        tmp_path
        / f"tt_{model['test_size']}_val_{model['valid_size']}_rescale_{scaling}_gamma_{model['gamma']}"
        / "model.jolib"
    )
    assert out_file.exists()


# Test if the model learns anything at all.
def test_small_data_overfit_checking():
    # create splits
    X_train, X_test, X_valid, Y_train, Y_test, Y_valid = test_splits(0.2, 0.2)
    # create model
    model = plot.train_model(X_train, X_train, Y_train, Y_train, gamma=0.5)
    # assert performance above minimal threshold
    acc_threshold = 0.5
    f1_threshold = 0.5
    # test
    assert model["acc"] > acc_threshold
    assert model["f1"] > f1_threshold


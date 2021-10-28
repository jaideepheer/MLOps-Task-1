# Import datasets, classifiers and performance metrics
from joblib import dump, load
from sklearn import datasets, svm
import os, pathlib
from .utils import *


def write_trained_model(model_dict, models_root: pathlib.Path):
    out_folder = models_root / serialize_dict_to_filename(model_dict)
    if out_folder.exists():
        print(f"Model folder ({out_folder}) exists. Overwriting...")
        os.system(f"rm -rf {str(out_folder.resolve())}")
    os.mkdir(out_folder)
    # save
    dump(model_dict["model"], out_folder / "model.jolib")


# Run train
if __name__ == "__main__":
    digits = datasets.load_digits()
    print(f"Original image size: {digits.images.shape}")
    data = digits.images
    n_samples = len(data)
    # set default models root
    models_root = pathlib.Path("./models")
    if models_root.exists():
        print(f"Models folder ({models_root}) exists.")
        if (input(f"Do you want to overwrite data?[Y/n]:") or "y").lower() == "y":
            os.system(f"rm -rf {str(models_root.resolve())}")
    os.mkdir(models_root)
    # main hparams
    hparam_space = {
        'scaling': [0.25, 0.5, 1, 2, 3],
        'sizes': [(0.15, 0.15), (0.2, 0.1)],
    }
    # choose classifier
    classifier = svm.SVC
    model_hparam_space = {
        "gamma": [10 ** exp for exp in range(-7, 0)],
    }
    # Loop over hparams
    for hparams in param_grid_iterator(hparam_space):
        # rescale
        images = preprocess_images(digits.images, hparams["scaling"])
        # flatten the images
        data = images.reshape((n_samples, -1))
        # Split data into 50% train and 50% test subsets
        test_size, valid_size = hparams["sizes"]
        X_train, X_test, X_valid, Y_train, Y_test, Y_valid = create_ttv_splits(
            data, digits.target, test_size, valid_size
        )
        # search hparams
        models = list()
        for m_params, model_dict in hparam_search_model(
            X_train,
            X_valid,
            Y_train,
            Y_valid,
            model_builder=svm.SVC,
            model_hparams=model_hparam_space,
        ):
            model_dict = {
                **hparams,
                **m_params,
                **model_dict,
            }
            # skip weak models
            if model_dict["valid_metrics"]["acc"] < 0.11:
                print(f"Skipping for {hparams} / {m_params}.")
                continue
            else:
                write_trained_model(
                    model_dict, models_root,
                )

            # save model
            models.append(model_dict)
            
        # predict
        max_f1_model = max(models, key=lambda x: x["valid_metrics"]["f1"])
        best_folder = models_root / serialize_dict_to_filename(max_f1_model)
        # load best
        clf = load(best_folder / "model.jolib")
        # metrics
        model_dict["test_metrics"] = predict_metrics(clf, X_test, Y_test)
        # print
        model_stats = serialize_dict_to_filename(
            model_dict,
            path_seperator="/",
            key_value_seperator=":",
            item_seperator="\t",
        )
        print(f"{images.shape[-2]}x{images.shape[-1]}\t{model_stats}")

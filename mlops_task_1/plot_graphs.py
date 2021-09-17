# Import datasets, classifiers and performance metrics
from joblib import dump, load
from sklearn import datasets, svm
import os, pathlib
from utils import *

digits = datasets.load_digits()

print(f"Original image size: {digits.images.shape}")

data = digits.images
n_samples = len(data)

# Make checkpoint root dir
models_root = pathlib.Path("./models")
if models_root.exists():
    print(f"Models folder ({models_root}) exists.")
    if (input(f"Do you want to overwrite data?[Y/n]:") or "y").lower() == "y":
        os.system(f"rm -rf {str(models_root.resolve())}")
os.mkdir(models_root)

# Run train
for test_size, valid_size in [(0.15, 0.15), (0.2, 0.1)]:
    for scaling in [0.25, 0.5, 1, 2, 3]:
        models = []
        for gamma in [10 ** exp for exp in range(-7, 0)]:
            # resize data
            images = preprocess_images(digits.images, scaling)

            # flatten the images
            data = images.reshape((n_samples, -1))

            # Create a classifier: a support vector classifier
            clf = svm.SVC(gamma=gamma)

            # Split data into 50% train and 50% test subsets
            X_train, X_test, X_valid, Y_train, Y_test, Y_valid = create_ttv_splits(
                data, digits.target, test_size, valid_size
            )

            # Learn the digits on the train subset
            clf.fit(X_train, Y_train)

            # test model
            valid_metrics = predict_metrics(clf, X_valid, Y_valid)

            # skip weak models
            if valid_metrics["acc"] < 0.11:
                print(f"Skipping for gamma={gamma}.")
                continue

            model = {
                **valid_metrics,
                "gamma": gamma,
            }
            models.append(model)
            # print(f"{gamma:.3f} --> {images.shape[1:]} --> {1-test_size}/{test_size} --> {acc:.3f} --> {f1:.3f}")
            out_folder = (
                models_root
                / f"tt_{test_size}_val_{valid_size}_rescale_{scaling}_gamma_{gamma}"
            )
            os.mkdir(out_folder)
            # save
            dump(clf, out_folder / "model.jolib")

        # predict
        max_f1_model = max(models, key=lambda x: x["f1"])
        best_folder = (
            models_root
            / f"tt_{test_size}_val_{valid_size}_rescale_{scaling}_gamma_{max_f1_model['gamma']}"
        )
        # load best
        clf = load(best_folder / "model.jolib")
        # metrics
        metrics = predict_metrics(clf, X_valid, Y_valid)
        # print
        print(
            f"{images.shape[-2]}x{images.shape[-1]}\t{max_f1_model['gamma']}\t{(1-test_size)*100}\t{test_size*100}\t{metrics['acc']}\t{metrics['f1']}"
        )


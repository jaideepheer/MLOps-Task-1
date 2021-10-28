from sklearn import svm, tree, datasets
from .utils import *

# Run train
if __name__ == "__main__":
    # load dataset
    digits = datasets.load_digits()
    print(f"Original image size: {digits.images.shape}")
    data = digits.images
    n_samples = len(data)
    # main hparams
    hparam_space = {
        "scaling": [0.25, 0.5, 1, 2, 3],
        "sizes": [(0.1, 0.1), (0.15, 0.15), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4)],
    }
    models = [
        {"builder": svm.SVC, "hparams": {"gamma": 0.1,},},
        {"builder": tree.DecisionTreeClassifier, "hparams": {"max_depth": 5,},},
    ]
    # iterate over hparam space
    for hparams in param_grid_iterator(hparam_space):
        print('='*15)
        print(f"Train/Test splits: {hparams['sizes']}")
        print('='*15)
        # rescale
        images = preprocess_images(digits.images, hparams["scaling"])
        # flatten the images
        data = images.reshape((n_samples, -1))
        # Split data into train and test subsets
        test_size, valid_size = hparams["sizes"]
        X_train, X_test, X_valid, Y_train, Y_test, Y_valid = create_ttv_splits(
            data, digits.target, test_size, valid_size
        )
        # train models 5 times with same split
        temp = '\t\t\t'.join(map(lambda k: str(k['builder'].__name__), models))
        print(f"Accuracy\t{temp}")
        total = []
        for i in range(5):
            res = [train_model(X_train, X_valid, Y_train, Y_valid, model_builder=m["builder"], **m["hparams"]) for m in models]
            res = list(map(lambda k: k["valid_metrics"]["acc"], res))
            total.append(res)
            temp = '\t'.join(map(str, res))
            print(f"        \t{temp}")
        v1, v2 = zip(*total)
        m1, m2 = sum(v1)/len(v1), sum(v2)/len(v2)
        print(f"Mean     \t{m1}\t{m2}")

        # print(f"Std. Dev.\t{s1}\t{s2}")
            

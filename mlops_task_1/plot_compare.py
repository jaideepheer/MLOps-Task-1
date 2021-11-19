from numpy.lib.function_base import append
from sklearn import svm, tree, datasets
from .utils import *
import pathlib
from joblib import dump, load

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
        {"builder": svm.SVC, "hparam_space": {
            "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, ]}, },
        {"builder": tree.DecisionTreeClassifier,
            "hparam_space": {"max_depth": [5, 6, 7, 8, 9, ], }, },
    ]
    # iterate over hparam space
    for hparams in param_grid_iterator(hparam_space):
        print('='*15)
        print(f"Scaling: {hparams['scaling']}")
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
            def _best_acc(builder, hparam_space):
                ll = []
                for hparams in param_grid_iterator(hparam_space):
                    ll.append(train_model(X_train, X_valid, Y_train, Y_valid,
                                          model_builder=builder, **hparams))
                return max(ll, key=lambda x: x["valid_metrics"]["acc"])
            res = [_best_acc(m["builder"], m["hparam_space"]) for m in models]
            total.append(res)
            res_acc = list(map(lambda k: k["valid_metrics"]["acc"], res))
            temp = '\t'.join(map(str, res_acc))
            print(f"        \t{temp}")
        total_acc = [list(map(lambda k: k["valid_metrics"]["acc"], x))
                     for x in total]
        v1, v2 = zip(*total_acc)
        m1, m2 = sum(v1)/len(v1), sum(v2)/len(v2)
        print(f"Mean     \t{m1}\t{m2}")

        def _stddev(l, mn):
            s = 0
            for x in l:
                s += (x-mn)**2
            return (s/len(l))**0.5
        s1, s2 = _stddev(v1, m1), _stddev(v2, m2)
        print(f"Std. Dev.\t{s1}\t{s2}")

        # Save best models
        mdls = zip(*total)
        sp = 'x'.join(map(str, list(images.shape[1:])))
        root = pathlib.Path(f'./models/best_{sp}')
        root.mkdir(parents=True, exist_ok=True)
        for m in mdls:
            builder = m[0]['model_builder'].__name__
            best = max(m, key=lambda x: x["valid_metrics"]["acc"])
            mfile = root / f'{builder}.gz'
            tosave = True
            if mfile.exists():
                # check if prev. model was better
                prev = load(mfile)
                if prev["valid_metrics"]["acc"] > best["valid_metrics"]["acc"]:
                    tosave = False
            if tosave is True:
                # new one is better
                dump(best, mfile)

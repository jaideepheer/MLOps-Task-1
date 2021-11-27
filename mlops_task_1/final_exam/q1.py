from sklearn import datasets, svm
from addict import Dict as D
from ..utils import create_ttv_splits, param_grid_iterator, predict_metrics, flatten_dict
import json
from tqdm.auto import tqdm

if __name__ == "__main__":
    digits = datasets.load_digits()
    print(f"Original image size: {digits.images.shape}")
    data = digits.images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    # set default model
    model = svm.SVC
    print(f"Classifier: {model}")
    hparam_space = {
        "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, ],
    }
    stats = []
    for hparams in tqdm(param_grid_iterator(hparam_space)):
        metrics = []
        for i in tqdm(range(3)):
            # Shuffle is true by default
            X_train, X_test, X_valid, Y_train, Y_test, Y_valid = create_ttv_splits(
                data, digits.target, test_size=0.15, valid_size=0.15,
            )
            # fit model
            m = model(**hparams)
            m.fit(X_train, Y_train)
            # collect metrics
            metrics.append(D({
                'train': predict_metrics(m, X_train, Y_train),
                'valid': predict_metrics(m, X_valid, Y_valid),
                'test': predict_metrics(m, X_test, Y_test),
            }))

        def _mean(m):
            ag = D()
            n = len(m)
            for mt in m:
                for dt in flatten_dict(mt):
                    k = tuple(dt[:-1])
                    v = dt[-1]
                    if k in ag:
                        ag[k] += v
                    else:
                        ag[k] = v
            # calc mean
            ag = {'/'.join(k): (v/n) for k, v in ag.items()}
            return ag
        stats.append(D({
            'hparams': hparams,
            'metrics': metrics,
            'mean_metrics': _mean(metrics),
        }))
    print(json.dumps(stats, sort_keys=True, indent=4,))

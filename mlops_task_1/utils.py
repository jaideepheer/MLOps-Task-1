from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

def preprocess_images(images, rescale_factor, anti_aliasing=False):
    return np.array([rescale(x, rescale_factor, anti_aliasing=anti_aliasing) for x in images])

def create_ttv_splits(data, target, test_size, valid_size):
    X_train, X_test, Y_train, Y_test = train_test_split(
        data, target, test_size=round((test_size+valid_size)*len(data)), shuffle=False)
    X_test, X_valid, Y_test, Y_valid = train_test_split(
        X_test, Y_test,
        test_size=round(valid_size*len(data)),
        shuffle=False,
    )
    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid

def predict_metrics(model, X, Y):
    predicted = model.predict(X)
    acc = metrics.accuracy_score(Y, predicted)
    f1 = metrics.f1_score(Y, predicted, average='macro')
    return {
        'model': model,
        'acc': acc,
        'f1': f1,
    }
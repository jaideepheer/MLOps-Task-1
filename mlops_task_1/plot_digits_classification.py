# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize
from sklearn.utils import resample

digits = datasets.load_digits()

print(f"Original image size: {digits.images.shape}")

data = digits.images

def print_metrics(images, test_size, gamma=1e-3):
    # flatten the images
    n_samples = len(images)
    data = images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=test_size, shuffle=False)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, predicted)
    f1 = metrics.f1_score(y_test, predicted, average='micro')
    print(f"{gamma:.3f} --> {images.shape[1:]} --> {1-test_size}/{test_size} --> {acc:.3f} --> {f1:.3f}")

print("Gamma --> Image_Size --> train/test --> Accuracy --> F1 Score (micro)")
for gamma in (1e-3, 1e-2, 0.1, 0.2, 0.5, 0.8, 1):
    for sz in (64, 32, 16):
        # resize data
        images = resize(data, (data.shape[0] ,sz, sz), anti_aliasing=True)
        for tsplit in (0.1, 0.2, 0.3, 0.4):
            print_metrics(images, tsplit, gamma)
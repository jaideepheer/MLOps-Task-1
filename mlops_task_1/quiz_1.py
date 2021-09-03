# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize
from sklearn.utils import resample

digits = datasets.load_digits()

print(f"Original image size: {digits.images.shape}")

data = digits.images

def print_metrics(images, test_size):
    # flatten the images
    n_samples = len(images)
    data = images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=test_size, shuffle=False)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, predicted)
    print(f"{images.shape[1:]} --> {1-test_size}/{test_size} --> {acc}")

print("Image_Size --> train/test --> Accuracy")
for sz in (64, 32, 16):
    # resize data
    images = resize(data, (data.shape[0] ,sz, sz), anti_aliasing=True)
    for tsplit in (0.1, 0.2, 0.3, 0.4):
        print_metrics(images, tsplit)
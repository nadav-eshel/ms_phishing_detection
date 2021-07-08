import numpy as np
import preprocessing
from sklearn.ensemble import RandomForestClassifier as rfc
# from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def predict_url(url, path_to_data_set):
    # Importing dataset
    # data = np.loadtxt(path_to_data_set + 'dataset.csv', delimiter=",")
    data = np.loadtxt(path_to_data_set + 'dataset_example.csv', delimiter=",")  # learn from example_data

    # Separating features and labels
    X = data[:, :-1]
    y = data[:, -1]
    # Separating to train:test with 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = rfc()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    # print(score*100)
    X_new = []
    X_input = url
    X_new = preprocessing.extract_features(X_input)
    X_new = np.array(X_new).reshape(1, -1)

    prediction = clf.predict(X_new)
    if prediction == -1:
        return "Phishing Url"
    else:
        return "Legitimate Url"


def main(path_to_urls, url, path_to_data_set):
    # # Create a feature based data set from url lists
    # preprocessing.create_data_set(path_to_urls, path_to_data_set, mode='phishing')
    # preprocessing.create_data_set(path_to_urls, path_to_data_set, mode='benign')

    # Run prediction on sample
    print(predict_url(url, path_to_data_set))


if __name__ == "__main__":

    path_to_urls = '/home/nadav/PycharmProjects/perception_point_nadav/files/urls.txt'
    url = 'http://matixlogin.eshost.com.ar/'
    path_to_data_set = '/home/nadav/PycharmProjects/perception_point_nadav/files/'

    main(path_to_urls, url, path_to_data_set)

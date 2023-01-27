import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


def labels_to_numeric(labels_df):
    # 0 - BE, 1 - CA, 2 - CH, 3 - FR
    labels_df["Country"] = labels_df["Country"].replace({'BE': 0})
    labels_df["Country"] = labels_df["Country"].replace({'CA': 1})
    labels_df["Country"] = labels_df["Country"].replace({'CH': 2})
    labels_df["Country"] = labels_df["Country"].replace({'FR': 3})

    print(np.array(labels_df.values).flatten())

    return list(np.array(labels_df.values).flatten())


def load_data(data_dir, feats_fname, labels_fname, scope):
    # Paths
    feats_path = os.path.join(data_dir, feats_fname)
    labels_path = os.path.join(data_dir, labels_fname)

    # Load features
    features = np.loadtxt(feats_path, delimiter=',')
    print(scope, " features shape: ", features.shape)

    # Load labels
    labels_df = pd.read_csv(labels_path)
    labels = labels_to_numeric(labels_df)
    print(scope, " labels length: ", len(labels))

    return features, labels


# method to write the predictions in the proper format
def writePredictions(predictions, file_path):
    d = {0: "BE", 1: "CA", 2: "CH", 3: "FR"}
    preds = [d[elem] for elem in predictions]
    df = pd.DataFrame(preds, columns=["Country"])
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    # Data directory
    model_file="svm_model.joblib"
    data_dir = "../data/bert_embeddings/"
    # Load the data
    #train_features, train_labels = load_data(data_dir, "train_embeddings.csv", "train_labels.txt", "Train")
    val_features, val_labels = load_data(data_dir, "val_embeddings.csv", "val_labels.txt", "Validation")
    test_features, test_labels = load_data(data_dir, "test_embeddings.csv", "test_labels.txt", "Test")

    # Test data
    clf = joblib.load(model_file)

    # Test/Predict
    print("TEST data:")
    y_pred_test = clf.predict(test_features)
    print(confusion_matrix(test_labels, y_pred_test))
    print(classification_report(test_labels, y_pred_test, digits=6, target_names=["BE", "CA", "CH" ,"FR"]))


    # Validation data
    print("VAL data:")
    y_pred_val = clf.predict(val_features) 
    print(confusion_matrix(val_labels, y_pred_val))
    print(classification_report(val_labels, y_pred_val, digits=6, target_names=["BE", "CA", "CH" ,"FR"]))


    writePredictions(y_pred_test, os.path.join(".", "svm_preds_test.csv"))
    writePredictions(y_pred_val, os.path.join(".", "svm_preds_val.csv"))


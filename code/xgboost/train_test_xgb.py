import os
import time
import joblib
import numpy as np
import pandas as pd

import xgboost as xgb
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


def fine_tune_xgb(X_train, y_train, model_fname):
    # Initialize the classifier
    clf = xgb.XGBClassifier(
        max_depth=200,
        n_estimators=400,
        subsamples=1,
        learning_rate=0.07,
        reg_lambda=0.1,
        reg_alpha=0.1,
        gamma=1)


    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print("======> Elapsed time for training with one set of parameters: %.10f" % (end - start))

    # Save model
    joblib.dump(clf, model_fname)

    return clf


if __name__ == "__main__":
    # Data directory
    data_dir = "../data/bert_embeddings/"
    # Load the data
    train_features, train_labels = load_data(data_dir, "train_embeddings.csv", "train_labels.txt", "Train")
    val_features, val_labels = load_data(data_dir, "val_embeddings.csv", "val_labels.txt", "Validation")
    test_features, test_labels = load_data(data_dir, "test_embeddings.csv", "test_labels.txt", "Test")

    # Fine tune
    clf = fine_tune_xgb(train_features, train_labels, "xgb_model.joblib")

    
    # Test
    test_preds = clf.predict(test_features)
    print("Test results:")
    print(confusion_matrix(test_labels, test_preds))
    print(classification_report(test_labels, test_preds, digits=6, target_names=["BE", "CA", "CH" ,"FR"]))

    # Validation
    val_preds = clf.predict(val_features)
    print("Validation results:")
    print(confusion_matrix(val_labels, val_preds))
    print(classification_report(val_labels, val_preds, digits=6, target_names=["BE", "CA", "CH" ,"FR"]))



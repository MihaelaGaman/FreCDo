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


def fine_tune_svm(X_train, y_train, model_fname):
    # Parameters
    param_grid = {
        'C': [0.0001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        }

    # Initialize the classifier
    clf = LinearSVC()#(probability=True)

    # Metrics
    metrics_list = ['accuracy', 'f1_macro', 'f1_weighted', \
                    'precision_macro', 'precision_weighted', \
                    'recall_macro', 'recall_weighted']

    cv = GridSearchCV(clf, param_grid, cv=10, verbose=2, \
                      scoring=metrics_list, refit='accuracy', \
                      return_train_score=True)

    start = time.time()
    cv.fit(X_train, y_train)
    end = time.time()
    print("======> Elapsed time for training with one set of parameters: %.10f" % (end - start))

    print("Best parameters: ", cv.best_params_)
    print("Grid scores on development set: ")
    for score_name in metrics_list:
        print("mean_score %s is %s" % (score_name, str(cv.cv_results_['mean_test_' + score_name])))

    # Save model
    joblib.dump(cv, model_fname)

    print(cv.best_estimator_)

    return cv


if __name__ == "__main__":
    # Data directory
    data_dir = "../data/bert_embeddings/"
    # Load the data
    train_features, train_labels = load_data(data_dir, "train_embeddings.csv", "train_labels.txt", "Train")
    #val_features, val_labels = load_data(data_dir, "val_embeddings.csv", "val_labels.txt", "Validation")
    test_features, test_labels = load_data(data_dir, "test_embeddings.csv", "test_labels.txt", "Test")

    # Fine tune
    grid = fine_tune_svm(train_features, train_labels, "svm_model.joblib")

    
    # Test
    grid_preds = grid.predict(test_features)
    print(confusion_matrix(test_labels, grid_preds))
    print(classification_report(test_labels, grid_preds, digits=6, target_names=["BE", "CA", "CH" ,"FR"]))


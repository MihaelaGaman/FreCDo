import csv
import os
import numpy as np
import pandas as pd


def loadData(purpose, fpath, has_labels=True):
    print("Reading labels from: ", fpath)
    print("Loading %s data..." % purpose)
    #return lat, longi
    if has_labels:
        return parseFileWithLabel(fpath)

    return parseFileWithoutLabel(fpath)

# parse training and validation files
def parseFileWithLabel(file_path):
    with open(file_path, "r") as f:
        data = f.read().splitlines()
        features = [splitted_line[1] for splitted_line in
                    [line.split("\t", maxsplit=1) for line in data[1:]]]
        labels = np.array([splitted_line[0] for splitted_line in
                [line.split("\t", maxsplit=1) for line in data[1:]]])
        
        print("Labels shape: ", labels.shape)
        print("Samples length: ", len(features))

        return features, labels #list(labels[:, 0]), list(labels[:, 1])

# parse test file
def parseFileWithoutLabel(file_path):
    with open(file_path, "r") as f:
        data = f.read().splitlines()
        features = [splitted_line[0] for splitted_line in
                    [line.split("\t", maxsplit=1) for line in data[1:]]]
    return features


def labels_to_numeric(labels):
    # Data frame from list
    labels_df = pd.DataFrame(labels)

    # 0 - BE, 1 - CA, 2 - CH, 3 - FR
    labels_df[0] = labels_df[0].replace({'BE': 0})
    labels_df[0] = labels_df[0].replace({'CA': 1})
    labels_df[0] = labels_df[0].replace({'CH': 2})
    labels_df[0] = labels_df[0].replace({'FR': 3})

    print(np.array(labels_df.values).flatten())

    return list(np.array(labels_df.values).flatten())


def flatten_labels(true_labels, predictions):
    true_labels_flat = []
    predictions_flat = []
    for index in range(len(true_labels)):
        true_labels_flat += list(true_labels[index])
        pred_flat = np.argmax(predictions[index], axis=1).flatten()
        predictions_flat += list(pred_flat)

    #print(true_labels_flat)
    #print(predictions_flat)

    return true_labels_flat, predictions_flat

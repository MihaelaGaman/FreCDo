import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import torch
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, classification_report
from transformers import CamembertModel, CamembertTokenizer, CamembertForSequenceClassification

import pandas as pd
import numpy as np

from loadDataSet import loadData, labels_to_numeric, flatten_labels
from helpers import get_device


def get_prediction(text, tokenizer, model, max_len, device):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    #print(outputs[1])
    # executing argmax function to get the candidate label
    return probs#.argmax()


if __name__ == "__main__":
    # Device
    device = get_device()

    # Paths
    base_path = "../code/"#"/home/mgaman/projects/french_dialect/data/Corpus/"
    test_path = base_path + "test_slices.txt"

    # Load the data
    testSamples, testLabels = loadData("test", test_path)

    print("Test size: %d" % len(testSamples))

    # BERT directory
    bert_dir = './bert_models_saved/best_model/'
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = CamembertTokenizer.from_pretrained(bert_dir, do_lowercase=True)

    # Compute max sentence length
    # Sentence length: Max =  512; Min =  3; Average = 445.26780711145534
    #compute_max_sent_length(tokenizer, trainSamples)

    # Model path
    model_path = os.path.join(bert_dir, "pytorch_model.bin")

    # Load the model
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=4, output_hidden_states=True)
    model = model.float()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Use approx the average in the dataset
    max_len = 128

    # Labels to numeric format
    # 0 - BE, 1 - CA, 2 - CH, 3 - FR
    testLabels = labels_to_numeric(testLabels)
 
    # Predict
    preds_all_labels = []
    preds_proba = []
    preds = []

    for i in range(len(testSamples)):
        # Get prediction
        pred = get_prediction(testSamples[1], tokenizer, model, max_len, device)
        # Get the list of pred probs for each class
        pred_list = pred.cpu().detach().numpy().tolist()[0]
        preds_all_labels.append(pred_list)
        # Get the prediction probability for the most likely class
        pred_proba = pred.max().item()
        preds_proba.append(pred_proba)
        # Get the predicted class index
        pred_index = pred.argmax().item()
        preds.append(pred_index)

    print(classification_report(testLabels, preds, digits=6, target_names=["BE", "CA", "CH", "FR"]))
 

    batch_size = 32

    # Model class must be defined somewhere
    model_path = os.path.join(bert_dir, "bert.model")

    best_model = CustomBERTModel()
    

    # Predict
    test_gt, test_preds = predict(best_model, input_ids_test, test_dataloader, device)

    # Accuracy
    compute_accuracy(test_gt, test_preds)

    # Flatten
    test_gt_flat, test_preds_flat = flatten_labels(test_gt, test_preds)

    # Macro F1 score
    f1_macro = f1_score(test_gt_flat, test_preds_flat, average='macro')
    print("F1 macro: ", f1_macro)

    

    # Write preds in file
    #writePredictions(train_preds, os.path.join(".", "predictions_train.csv"))
    #writePredictions(val_preds, os.path.join(".", "predictions_validation.csv"))
    #writePredictions(test_preds, os.path.join(".", "predictions_test.csv"))


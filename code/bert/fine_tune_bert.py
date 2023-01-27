import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from transformers import CamembertTokenizer, CamembertForSequenceClassification, CamembertConfig
from transformers import Trainer, TrainingArguments

import pandas as pd
import numpy as np

from loadDataSet import loadData, labels_to_numeric
from helpers import compute_max_sent_length, get_device, set_seed
from bert_utils import (
    FrenchDataset,
    compute_metrics,
)

from nltk.tokenize import sent_tokenize

set_seed(1)

if __name__ == "__main__":
    # Device
    device = get_device()

    # Paths
    base_path = "../code/"#"/home/mgaman/projects/french_dialect/data/Corpus/"
    train_path = base_path + "train_slices.txt"
    val_path = base_path + "val_slices.txt"

    # Load the data
    trainSamples, trainLabels = loadData("train", train_path)
    valSamples, valLabels = loadData("validation", val_path)

    print("Initial train size: %d" % len(trainSamples))
    print("Val size: %d" % len(valSamples))

    # Load the CamemBERT tokenizer
    print("Loading CamemBERT tokenizer...")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # Compute max sentence length
    # Sentence length: Max =  512; Min =  3; Average = 445.26780711145534
    # For 3-sentence paragraphs: Average sentence length:  102.64167598045637
    #compute_max_sent_length(tokenizer, trainSamples)


    # Use approx the average in the dataset
    max_len = 128

    # Labels to numeric format
    # 0 - BE, 1 - CA, 2 - CH, 3 - FR
    trainLabels = labels_to_numeric(trainLabels)
    valLabels = labels_to_numeric(valLabels)


    # Tokenize / Prepare the training set
    train_encodings = tokenizer(trainSamples, truncation=True, padding=True, max_length=max_len)
    # Tokenize / Prepare the validation set
    valid_encodings = tokenizer(valSamples, truncation=True, padding=True, max_length=max_len)
    
    # Convert our tokenized data into a torch Dataset
    train_dataset = FrenchDataset(train_encodings, trainLabels)
    valid_dataset = FrenchDataset(valid_encodings, valLabels)

    # Load the model and pass to device
    config = CamembertConfig.from_pretrained("camembert-base", output_hidden_states=True)
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=4).to(device)

    # Train args
    training_args = TrainingArguments(
        output_dir="./bert_models_saved/out_fold",          # output directory
        num_train_epochs=30,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=500,              # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=250,               # log & save weights each logging_steps
        eval_steps=250,
        #learning_rate=5e-5,
        save_total_limit=5,
        save_strategy="steps",
        evaluation_strategy="steps",     # evaluate each `logging_steps`
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

    # Train the model
    trainer.train()

    # Save best only
    trainer.save_model("./bert_models_saved/out_fold")

    # Evaluate the best performing model
    trainer.evaluate()

    # Save for later
    model.save_pretrained("./bert_models_saved/best_model/")
    tokenizer.save_pretrained("./bert_models_saved/best_model/")



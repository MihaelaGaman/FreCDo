import numpy as np
import random
import torch
import time
import datetime
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available


def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():    
    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
    
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
    
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    return device


def compute_max_sent_length(tokenizer, sentences):
    max_len = 0
    avg_len = 0
    min_len = 100000
    
    # For every sentence...
    for sent in sentences:
    
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(
            sent,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )
    
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

        # Min length
        min_len = min(min_len, len(input_ids))

        # Average
        avg_len += len(input_ids)

    avg_len = avg_len / len(sentences)
    
    print('Max sentence length: ', max_len)
    print('Min sentence length: ', min_len)
    print('Average sentence length: ', avg_len)

    return max_len


def print_model(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    
    print('==== Embedding Layer ====\n')
    
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    print('\n==== First Transformer ====\n')
    
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    print('\n==== Output Layer ====\n')
    
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


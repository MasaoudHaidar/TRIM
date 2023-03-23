# main libraries
import numpy as np
import spacy
import re

# specific machine learning functionality
import tensorflow as tf

# Transformers
from transformers import (
    BertTokenizer,
)

classifier_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(classifier_name, do_lower_case=True)
batch_size = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE
sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
to_keep = [
    "n't",
    "neither",
    "never",
    "no",
    'noone',
    'nor',
    'not',
    'nothing',
    'n‘t',
    'n’t',
    'only',
    'quite',
    'really',
    'serious',
    'several',
    'still',
    'such',
    'take',
    'too',
    'top',
    'unless',
    'various',
    'very',
    'well',
]
all_stopwords = [word for word in all_stopwords if not word in to_keep]
all_stopwords = set(all_stopwords)
letter_regex = re.compile('[^a-zA-Z]')
lasso_alpha = 0.01
top_k_words = 10


### Tokenization function
def __tokenize_for_bert_classifier(df, should_shuffle=False):
    # Tokenization
    X_tokenized = bert_tokenizer.batch_encode_plus(
        df["text"],
        return_tensors='tf',
        add_special_tokens=True,
        return_token_type_ids=True,
        padding='max_length',
        max_length=256,
        return_attention_mask=True,
        truncation='longest_first'
    )
    # Creating TF datasets
    dataset = tf.data.Dataset.from_tensor_slices(((X_tokenized["input_ids"],
                                                   X_tokenized["token_type_ids"],
                                                   X_tokenized["attention_mask"]),
                                                  df["label"]))
    if should_shuffle:
        buffer_train = len(df["text"])
        dataset = dataset.shuffle(buffer_size=buffer_train)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def __inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


def __format_color_style(word, score):
    if 0.1 > score > -0.1:
        return word
    elif 0.1 <= score < 0.8:
        return f'\x1b[1;37;46m {word} \x1b[0m'
    elif score >= 0.8:
        return f'\x1b[1;37;42m {word} \x1b[0m'
    elif -0.1 >= score > -0.8:
        return f'\x1b[1;37;45m  {word} \x1b[0m'
    else:
        return f'\x1b[1;37;41m  {word} \x1b[0m'


def __get_color(df):
    x = df.copy()
    for i, row in df.iterrows():
        if row["importance"] > 0:
            green_value = min(max(0, row["importance"]*80), 256)
            style = f'background-color: rgb({256 - green_value}, 256, {256 - green_value})'
        else:
            red_value = min(max(0, -row["importance"]*80), 256)
            style = f'background-color: rgb(256, {256 - red_value}, {256 - red_value})'
        x.iloc[i] = style
    return x

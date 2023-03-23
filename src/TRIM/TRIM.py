# main libraries
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.autonotebook import tqdm

# sklearn
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Lasso

# specific machine learning functionality
import tensorflow as tf

# Transformers
from transformers import TFBertForMaskedLM, BertTokenizer

# Util functions
from utils import (
    top_k_words,
    letter_regex,
    __tokenize_for_bert_classifier,
    lasso_alpha,
    all_stopwords,
    __inv_logit,
    __get_color,
    __format_color_style,
)

classifier_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(classifier_name, do_lower_case=True)
gap_untuned_model = TFBertForMaskedLM.from_pretrained("bert-base-uncased")


def get_multiple_replacement_scores(
        original_sentence,
        classifier,
        verbose=False,
        gap_filler=gap_untuned_model,
        n_samples_per_word=2,
        return_type="table",  # table, list, or both
        ignore_first_x_words=0,
):
    # original sentence
    inputs = bert_tokenizer(original_sentence, return_tensors="tf")
    logits = classifier(**inputs).logits
    original_sentence_score = logits[0, 0].numpy()
    if verbose:
        print(original_sentence)
        print(f"Original score: {original_sentence_score}")
        print()

    # modefied sentences
    all_words = original_sentence.split()
    n_samples = len(all_words) * n_samples_per_word
    word_scores = defaultdict(list)
    X = []
    # replacement_size = int(np.sqrt(len(all_words)) + 1)
    replacement_size = int(len(all_words) * 0.15 + 1)
    sentences = []
    for _ in tqdm(range(n_samples), total=n_samples):
        # Sample masking indices
        word_indices = np.random.choice(
            range(ignore_first_x_words, len(all_words)),
            size=replacement_size,
            replace=False,
        )
        current_x_row = np.ones(len(all_words))
        for i in word_indices:
            current_x_row[i] = 0
        for _ in range(top_k_words):
            X.append(current_x_row)
        words = [all_words[i] for i in word_indices]
        words = [letter_regex.sub("", word).lower() for word in words]
        new_sentence = " ".join(
            [temp_word if j not in word_indices else "[MASK]" for (j, temp_word) in enumerate(all_words)])

        # get gap filler logits
        inputs = bert_tokenizer(new_sentence, return_tensors="tf")
        logits = gap_filler(**inputs).logits

        # retrieve indices of [MASK]
        mask_token_index = tf.where((inputs.input_ids == bert_tokenizer.mask_token_id)[0])
        selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

        # get top predictions
        predicted_token_ids = [tf.math.top_k(temp, top_k_words).indices for temp in selected_logits]
        options = [bert_tokenizer.decode(temp) for temp in predicted_token_ids]
        options = [temp.split() for temp in options]
        options = [temp if len(temp) == top_k_words
                   else temp + ["" for _ in range(top_k_words - len(temp))]
                   for temp in options]

        # get scores of those predictions
        filled_sentences = [new_sentence for _ in range(top_k_words)]
        for i in range(top_k_words):
            for j in range(replacement_size):
                filled_sentences[i] = filled_sentences[i].replace("[MASK]", options[j][i], 1)
            sentences.append(filled_sentences[i])

    # compute model outcomes:
    dataset = __tokenize_for_bert_classifier(
        pd.DataFrame({
            "text": sentences,
            "label": [True for _ in sentences]
        })
    )
    Y = classifier.predict(dataset).logits

    # Train a simple model on the local data
    simple_model = Lasso(lasso_alpha).fit(X, Y)

    if return_type == "table" or return_type == "both":
        filtered_words = list(filter(lambda w: w.lower() not in all_stopwords, all_words))
        all_words_unique = [letter_regex.sub("", word).lower() for word in filtered_words]
        all_words_unique = list(set(all_words_unique))
        word_importance_raw = defaultdict(list)
        for i, word in enumerate(all_words):
            word_importance_raw[letter_regex.sub("", word).lower()].append(simple_model.coef_[i])
        word_importance_df = pd.DataFrame(
            {
                "word": all_words_unique,
                "importance": [np.mean(word_importance_raw[temp]) for temp in all_words_unique]
            }
        )
        word_importance_df = word_importance_df.sort_values(by="importance", ignore_index=True)

    if return_type == "list" or return_type == "both":
        word_importance_list = []
        for i, word in enumerate(all_words):
            word_importance_list.append(simple_model.coef_[i])

    if verbose:
        print(f"Selection rates: {np.mean(X, axis=0)}")
        print(f"Outcome mean: {np.mean(Y):0.4f}")
        print(f"Model MSE: {simple_model.score(X, Y):0.4f}")
    print(f"Model MAPE: {mean_absolute_percentage_error(Y, simple_model.predict(X)):0.4f}")

    if return_type == "table":
        return word_importance_df
    elif return_type == "list":
        return all_words, word_importance_list
    else:
        return word_importance_df, (all_words, word_importance_list)


def show_multi_replacement_scores(
    original_sentence,
    classifier,
    verbose=False,
    gap_filler=gap_untuned_model,
    show_colored_text=False,
    replacement_sample_size=2,
    ignore_first_x_words=0,
  ):
    inputs = bert_tokenizer(original_sentence, return_tensors="tf")
    logits = classifier(**inputs).logits
    original_sentence_score = logits[0,0].numpy()
    print(f"Sentence score: {__inv_logit(original_sentence_score):0.4F}")
    if not show_colored_text:
        replacement_df = get_multiple_replacement_scores(
            original_sentence,
            classifier,
            verbose,
            gap_filler,
            replacement_sample_size,
            ignore_first_x_words)
        display(replacement_df.style.apply(__get_color, axis=None))
    else:
        replacement_df, (words, replacement_list) = get_multiple_replacement_scores(
            original_sentence,
            classifier,
            verbose,
            gap_filler,
            return_type="both",
            n_samples_per_word = replacement_sample_size,
            ignore_first_x_words = ignore_first_x_words,
        )
        display(replacement_df.style.apply(__get_color, axis=None))
        replacement_sentence = ' '.join([
            __format_color_style(word, score)
            for word, score in zip(words, replacement_list)
        ])
        print(replacement_sentence)
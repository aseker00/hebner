import errno
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from src.modeling.modeling_xfmr import XfmrNerModel


def get_io_seq_by_token(df: pd.DataFrame, agg_func, norm_func) -> (list, list):
    seq = [df.loc[df['sentence_id'] == sent_id].groupby('token_id').apply(agg_func) for sent_id in df.sentence_id.unique()]
    normalized_seq = [norm_func(s) for s in seq]
    tokens_seq = [[t for (t, l) in s] for s in normalized_seq]
    labels_seq = [[l for (t, l) in s] for s in normalized_seq]
    return tokens_seq, labels_seq


def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def set_pred_by_token(orig: pd.DataFrame, predictions: list, labels: list) -> pd.DataFrame:
    df = orig.copy()
    df['prediction'] = "O"
    for sent_id, sent_preds in enumerate(predictions):
        for token_id, pred_id in enumerate(sent_preds):
            label = labels[pred_id]
            if label[0] == "I":
                df.loc[(df["sentence_id"] == sent_id) & (df["token_id"] == token_id), "prediction"] = label
            elif label[0] == "B":
                df.loc[(df["sentence_id"] == sent_id) & (df["token_id"] == token_id), "prediction"] = label[0].replace('B', 'I') + label[1:]
                char_ids = list(df.loc[(df["sentence_id"] == sent_id) & (df["token_id"] == token_id), "char_id"])
                df.loc[(df["sentence_id"] == sent_id) & (df["token_id"] == token_id) & (df["char_id"] == char_ids[0]), "prediction"] = label
    return df


def to_sentences(df: pd.DataFrame) -> list:
    sentences = []
    gb = df.groupby('sentence_id')
    groups = dict(list(gb))
    for sent_id in groups:
        text = []
        char_ids = list(groups[sent_id]['char_id'])
        chars = list(groups[sent_id]['char'])
        i = 0
        for char_id, char in zip(char_ids, chars):
            while char_id > i:
                text.append(' ')
                i += 1
            text.append(char)
            i += 1
        sentences.append(''.join(text))
    return sentences


def to_labels(df: pd.DataFrame, label_col: str) -> list:
    sent_labels = []
    gb = df.groupby('sentence_id')
    groups = dict(list(gb))
    for sent_id in groups:
        labels = []
        char_ids = list(groups[sent_id]['char_id'])
        predictions = list(groups[sent_id][label_col])
        for i, (char_id, label) in enumerate(zip(char_ids, predictions)):
            if char_id > i:
                # if i + 1 < len(predictions):
                next_label = predictions[i]
                if next_label[0] == 'I':
                    labels.append(next_label)
                else:
                    labels.append('O')
            else:
                i += 1
            labels.append(label)
        sent_labels.append(labels)
    return sent_labels


def to_adms(df: pd.DataFrame, labels: list) -> list:
    adms = []
    sentences = to_sentences(df)
    for i, sent in enumerate(sentences):
        adm = {'attributes': {'entities': {'type': 'list', 'itemType': 'entities', 'items': []}}, 'data': sent}
        lst = labels[i]
        start_offset = 0
        end_offset = 0
        entity_type = "ORG"
        for a in lst:
            if a[0] == 'O':
                if end_offset > start_offset:
                    end_offset += 1
                    m = {'type': entity_type, 'mentions': [{'startOffset': start_offset, 'endOffset': end_offset}]}
                    adm['attributes']['entities']['items'].append(m)
                    # print(data[start_offset:end_offset])
                    start_offset = end_offset
                else:
                    start_offset += 1
            elif a[0] == 'B':
                end_offset = start_offset + 1
                if len(a) > 1:
                    entity_type = a[2:]
            elif a[0] == 'I':
                if end_offset < start_offset:
                    end_offset = start_offset + 1
                    if len(a) > 1:
                        entity_type = a[2:]
                else:
                    end_offset += 1
        adms.append(adm)
    return adms


def mkdir(folder_path: str):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def save_output(epoch: int, name: str, index: int, pred_adm: dict, gold_adm: dict):
    output_root_folder = 'outputs'
    output_folder_path = output_root_folder + "/e" + str(epoch)
    pred_output_path = output_folder_path + "/pred" + "/" + name
    gold_output_path = output_folder_path + "/gold" + "/" + name
    mkdir(pred_output_path)
    mkdir(gold_output_path)
    with open(pred_output_path + "/" + str(index) + '.adm.json', 'w') as outfile:
        json.dump(pred_adm, outfile)
    with open(gold_output_path + "/" + str(index) + '.adm.json', 'w') as outfile:
        json.dump(gold_adm, outfile)


# java -Xmx4g -jar corpuscmd-85.2.18.c61.0-stand-alone.jar CharLevelMucEval --referenceData outputs/h300-n2-test/e0/gold/ --testData outputs/h300-n2-test/e{}/pred/


def save_processed_dataset(dataset: pd.DataFrame, dataset_file_path: Path):
    dataset.to_csv(str(dataset_file_path))


def load_processed_dataset(dataset_file_path: Path) -> pd.DataFrame:
    return pd.read_csv(str(dataset_file_path))


def save_model_data_samples(dataset: pd.DataFrame, dataset_name: str, model: XfmrNerModel):
    gb = dataset.groupby('sent_idx')
    sequences = [(x, gb.get_group(x)) for x in gb.groups]
    max_seq_len = max([len(x[1].index) for x in sequences])
    data_samples = [model.to_sample(x[0], x[1], max_seq_len) for x in sequences]
    with open('data/processed/{}.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump(data_samples, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_data_samples(dataset_name) -> list:
    with open('data/processed/{}.pkl'.format(dataset_name), 'rb') as f:
        return pickle.load(f)


def is_english(s):
    try:
        # s.encode(encoding='utf-8').decode('ascii')
        s.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True


class LabeledSentence:

    def __init__(self, sent_id, tokens, labels):
        self.sent_id = sent_id
        self.tokens = tokens
        self.labels = labels

    def xfmr_data_row(self, token_idx: int, model: XfmrNerModel) -> dict:
        token = self.tokens[token_idx]
        label = self.labels[token_idx]
        label_id = model.label2id[label]
        xfmr_tokens = model.tokenizer.tokenize(token)
        xfmr_token_ids = model.tokenizer.convert_tokens_to_ids(xfmr_tokens)
        return {'sent_idx': [self.sent_id] * len(xfmr_tokens),
                'token_idx': [token_idx] * len(xfmr_tokens),
                'token': [token] * len(xfmr_tokens),
                'xfmr_token': xfmr_tokens,
                'xfmr_token_id': xfmr_token_ids,
                'label': [label] * len(xfmr_tokens),
                'label_id': [label_id] * len(xfmr_tokens)}

    def rex_data_row(self, token_idx: int, model: XfmrNerModel) -> dict:
        token = self.tokens[token_idx]
        label = self.labels[token_idx]
        label_id = model.label2id[label]
        return {'sent_idx': [self.sent_id],
                'token_idx': [token_idx],
                'token': [token],
                'label': [label],
                'label_id': [label_id]}


def process_labeled_sentences(labeled_sentences: list, model: XfmrNerModel) -> pd.DataFrame:
    cls = model.tokenizer.cls_token
    sep = model.tokenizer.sep_token
    # pad = model.tokenizer.pad_token
    sent_data = defaultdict(list)
    for sent in labeled_sentences:
        sent.tokens.insert(0, cls)
        sent.tokens.append(sep)
        # sent.tokens.append(pad)
        sent.labels.insert(0, cls)
        sent.labels.append(sep)
        # sent.labels.append(pad)
        sent_data_rows = [sent.xfmr_data_row(i, model) for i in range(len(sent.tokens))]
        for data_row in sent_data_rows:
            for k in data_row:
                sent_data[k].extend(data_row[k])
    return pd.DataFrame(sent_data)


def process_rex_labeled_sentences(labeled_sentences: list, model: XfmrNerModel) -> pd.DataFrame:
    sent_data = defaultdict(list)
    for sent in labeled_sentences:
        sent_data_rows = [sent.rex_data_row(i, model) for i in range(len(sent.tokens))]
        for data_row in sent_data_rows:
            for k in data_row:
                sent_data[k].extend(data_row[k])
    return pd.DataFrame(sent_data)

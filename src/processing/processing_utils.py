import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd
from src.modeling.modeling_xfmr import XfmrNerModel


def save_processed_dataset(dataset: pd.DataFrame, dataset_file_path: Path):
    dataset.to_csv(str(dataset_file_path))


def load_processed_dataset(dataset_file_path: Path) -> pd.DataFrame:
    return pd.read_csv(str(dataset_file_path))


def save_model_data_samples(labeled_sentences: list, dataset: pd.DataFrame, dataset_name: str, model: XfmrNerModel):
    labeled_sentences = {sent.sent_id: sent for sent in labeled_sentences}
    gb = dataset.groupby('sent_idx')
    sequences = [(x, labeled_sentences[x], gb.get_group(x)) for x in gb.groups]
    max_seq_len = max([len(x[2].index) for x in sequences])
    data_samples = [model.to_sample(x[0], x[1].text, x[2], max_seq_len) for x in sequences]
    with open('data/processed/{}-{}.pkl'.format(dataset_name, model.name), 'wb') as f:
        pickle.dump(data_samples, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_data_samples(dataset_name, model: XfmrNerModel) -> list:
    with open('data/processed/{}-{}.pkl'.format(dataset_name, model.name), 'rb') as f:
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

    def __init__(self, sent_id: int, text: str, token_offsets: dict, tokens: list, labels: list):
        self.sent_id = sent_id
        self.text = text
        self.offsets = [(token_start_offset, token_offsets[token_start_offset]) for token_start_offset in sorted(token_offsets)]
        self.tokens = tokens
        self.labels = labels

    def xfmr_data_row(self, token_idx: int, model: XfmrNerModel) -> dict:
        token = self.tokens[token_idx]
        token_offsets = self.offsets[token_idx]
        label = self.labels[token_idx]
        label_id = model.label2id[label]
        xfmr_tokens = model.tokenizer.tokenize(token)
        xfmr_token_ids = model.tokenizer.convert_tokens_to_ids(xfmr_tokens)
        return {'sent_idx': [self.sent_id] * len(xfmr_tokens),
                'token_idx': [token_idx] * len(xfmr_tokens),
                'token': [token] * len(xfmr_tokens),
                'token_start_offset': [token_offsets[0]] * len(xfmr_tokens),
                'token_end_offset': [token_offsets[1]] * len(xfmr_tokens),
                'xfmr_token': xfmr_tokens,
                'xfmr_token_id': xfmr_token_ids,
                'label': [label] * len(xfmr_tokens),
                'label_id': [label_id] * len(xfmr_tokens)}

    def rex_data_row(self, token_idx: int, model: XfmrNerModel) -> dict:
        token = self.tokens[token_idx]
        token_offsets = self.offsets[token_idx]
        label = self.labels[token_idx]
        label_id = model.label2id[label]
        return {'sent_idx': [self.sent_id],
                'token_idx': [token_idx],
                'token': [token],
                'token_start_offset': [token_offsets[0]],
                'token_end_offset': [token_offsets[1]],
                'label': [label],
                'label_id': [label_id]}

    def to_adm(self) -> dict:
        entity_items = []
        cur_label = 'O'
        cur_start_offset = 0
        cur_end_offset = 0
        for label, token_offset in zip(self.labels, self.offsets):
            if cur_label[0] == 'B' or cur_label[0] == 'I':
                if label[0] == 'I':
                    cur_end_offset = token_offset[1]
                else:
                    entity_item = {'type': cur_label[2:], 'mentions': [{'startOffset': int(cur_start_offset), 'endOffset': int(cur_end_offset)}]}
                    entity_items.append(entity_item)
                    cur_label = label
                    if label[0] == 'B':
                        cur_start_offset = token_offset[0]
                        cur_end_offset = token_offset[1]
            else:
                cur_label = label
                if label[0] == 'B' or label[0] == 'I':
                    cur_start_offset = token_offset[0]
                    cur_end_offset = token_offset[1]
        annotations = {'data': self.text, 'attributes': {'entities': {'type': 'list', 'itemType': 'entities', 'items': entity_items}}}
        return annotations


def process_labeled_sentences(labeled_sentences: list, model: XfmrNerModel) -> pd.DataFrame:
    cls = model.tokenizer.cls_token
    sep = model.tokenizer.sep_token
    # pad = model.tokenizer.pad_token
    sent_data = defaultdict(list)
    for sent in labeled_sentences:
        sent.offsets.insert(0, (-1, 0))
        sent.tokens.insert(0, cls)
        # sent.labels.insert(0, cls)
        sent.labels.insert(0, model.labels[0])
        sent.offsets.append((len(sent.text) - 1, len(sent.text)))
        sent.tokens.append(sep)
        # sent.labels.append(sep)
        sent.labels.append(model.labels[0])
        sent_data_rows = [sent.xfmr_data_row(i, model) for i in range(len(sent.tokens))]
        token_ids = [token_id for data_row in sent_data_rows for token_id in data_row['xfmr_token']]
        if len(token_ids) > model.max_seq_len:
            continue
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

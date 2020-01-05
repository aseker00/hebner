import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd
from src.modeling.modeling_xfmr import XfmrNerModel
from src.modeling.modeling_char_xfmr import CharXfmrNerModel


def save_processed_dataset(dataset: pd.DataFrame, dataset_file_path: Path):
    dataset.to_csv(str(dataset_file_path))


def load_processed_dataset(dataset_file_path: Path) -> pd.DataFrame:
    return pd.read_csv(str(dataset_file_path))


def save_model_data_samples(base_dir: str, labeled_sentences: list, token_df: pd.DataFrame, dataset_name: str,
                            x_model: XfmrNerModel):
    labeled_sentences = {sent.sent_id: sent for sent in labeled_sentences}
    token_gb = token_df.groupby('sent_idx')
    token_groups = {i: token_gb.get_group(i) for i in token_gb.groups}
    max_token_seq_len = max([len(token_groups[i].index) for i in token_groups])
    data_samples = [x_model.to_sample(i, labeled_sentences[i].text, token_groups[i], max_token_seq_len) for i in
                    token_groups]
    with open('{}/data/processed/{}-{}.pkl'.format(base_dir, dataset_name, x_model.name), 'wb') as f:
        pickle.dump(data_samples, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_data_samples(base_dir: str, dataset_name, model_name: str) -> list:
    with open('{}/data/processed/{}-{}.pkl'.format(base_dir, dataset_name, model_name), 'rb') as f:
        return pickle.load(f)


def save_char_model_data_samples(base_dir: str, labeled_sentences: list, token_df: pd.DataFrame, char_df: pd.DataFrame,
                                 dataset_name: str, cx_model: CharXfmrNerModel):
    # merged_df = pd.merge(token_df, char_df, on=['sent_idx', 'token_idx'])
    labeled_sentences = {sent.sent_id: sent for sent in labeled_sentences}
    token_gb = token_df.groupby('sent_idx')
    token_groups = {i: token_gb.get_group(i) for i in token_gb.groups}
    char_gb = char_df.groupby('sent_idx')
    char_groups = {i: char_gb.get_group(i) for i in char_gb.groups}
    # max_token_seq_len = max([len(token_groups[i].index) for i in token_groups])
    # max_char_seq_len = max([len(labeled_sentences[i].text) + 2 for i in token_groups])
    max_char_seq_len = max([len(char_groups[i].index) for i in char_groups])
    data_samples = [cx_model.to_sample(i, labeled_sentences[i].text, token_groups[i], char_groups[i], max_char_seq_len)
                    for i in token_groups]
    with open('{}/data/processed/{}-{}-{}.pkl'.format(base_dir, dataset_name, cx_model.name, 'char'), 'wb') as f:
        pickle.dump(data_samples, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def load_char_model_data_samples(base_dir: str, dataset_name, model_name: str) -> list:
    with open('{}/data/processed/{}-{}-{}.pkl'.format(base_dir, dataset_name, model_name, 'char'), 'rb') as f:
        return pickle.load(f)


def is_english(s):
    try:
        # s.encode(encoding='utf-8').decode('ascii')
        s.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True


class TokenLabeledSentence:

    def __init__(self, sent_id: int, text: str, token_offsets: dict, labels: list):
        self.sent_id = sent_id
        self.text = text
        self.token_offsets = [(token_start_offset, token_offsets[token_start_offset]) for token_start_offset in
                              sorted(token_offsets)]
        self.labels = labels

    @property
    def tokens(self) -> list:
        return [self.text[token_offset[0]:token_offset[1]] for token_offset in self.token_offsets]

    def xfmr_data_row(self, token_idx: int, x_model: XfmrNerModel) -> dict:
        token_offset = self.token_offsets[token_idx]
        if token_idx == 0:
            token = x_model.cls_token
            label = x_model.labels[0]
        elif token_idx == len(self.token_offsets) - 1:
            token = x_model.tokenizer.sep_token
            label = x_model.labels[0]
        else:
            token = self.text[token_offset[0]:token_offset[1]]
            label = self.labels[token_idx - 1]
        label_id = x_model.label2id[label]
        xfmr_tokens, xfmr_token_ids = x_model.tokenize(token)
        return {'sent_idx': [self.sent_id] * len(xfmr_tokens),
                'token_idx': [token_idx] * len(xfmr_tokens),
                'token': [token] * len(xfmr_tokens),
                'token_start_offset': [token_offset[0]] * len(xfmr_tokens),
                'token_end_offset': [token_offset[1]] * len(xfmr_tokens),
                'xfmr_token': xfmr_tokens,
                'xfmr_token_id': xfmr_token_ids,
                'token_label': [label] * len(xfmr_tokens),
                'token_label_id': [label_id] * len(xfmr_tokens)}

    def rex_data_row(self, token_idx: int, x_model: XfmrNerModel) -> dict:
        token_offset = self.token_offsets[token_idx]
        token = self.text[token_offset[0]:token_offset[1]]
        label = self.labels[token_idx]
        label_id = x_model.label2id[label]
        return {'sent_idx': [self.sent_id],
                'token_idx': [token_idx],
                'token': [token],
                'token_start_offset': [token_offset[0]],
                'token_end_offset': [token_offset[1]],
                'token_label': [label],
                'token_label_id': [label_id]}

    def to_adm(self) -> dict:
        entity_items = []
        cur_label = 'O'
        cur_start_offset = 0
        cur_end_offset = 0
        for label, token_offset in zip(self.labels, self.token_offsets):
            if cur_label[0] == 'B' or cur_label[0] == 'I':
                if label[0] == 'I':
                    cur_end_offset = token_offset[1]
                else:
                    entity_item = {'type': cur_label[2:], 'mentions': [{'startOffset': int(cur_start_offset),
                                                                        'endOffset': int(cur_end_offset)}]}
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
        annotations = {'data': self.text, 'attributes': {'entities': {'type': 'list', 'itemType': 'entities',
                                                                      'items': entity_items}}}
        return annotations


class CharLabeledSentence:

    def __init__(self, sent_id: int, text: str, token_offsets: dict, char_labels: list):
        self.sent_id = sent_id
        self.text = text
        self.token_offsets = [(token_start_offset, token_offsets[token_start_offset]) for token_start_offset in
                              sorted(token_offsets)]
        self.labels = char_labels

    @property
    def tokens(self) -> list:
        return [self.text[token_offset[0]:token_offset[1]] for token_offset in self.token_offsets]

    def xfmr_data_row(self, token_idx: int, cx_model: CharXfmrNerModel) -> dict:
        token_offset = self.token_offsets[token_idx]
        if token_idx == 0:
            chars = [cx_model.cls_token]
            labels = [cx_model.labels[0]]
        elif token_idx == len(self.token_offsets) - 1:
            chars = [cx_model.sep_token]
            labels = [cx_model.labels[0]]
        else:
            chars = list(self.text[token_offset[0]:token_offset[1]])
            labels = [self.labels[i] for i in range(token_offset[0], token_offset[1])]
        label_ids = [cx_model.label2id[label] for label in labels]
        char_ids = [cx_model.char2id[c] for c in chars]
        return {'sent_idx': [self.sent_id] * len(chars),
                'token_idx': [token_idx] * len(chars),
                'char': chars,
                'char_id': char_ids,
                'char_label': labels,
                'char_label_id': label_ids}


def process_xfmr_labeled_sentences(labeled_sentences: list, x_model: XfmrNerModel) -> pd.DataFrame:
    sent_data = defaultdict(list)
    for sent in labeled_sentences:
        sent.token_offsets.insert(0, (-1, 0))
        sent.token_offsets.append((len(sent.text) - 1, len(sent.text)))
        sent_data_rows = [sent.xfmr_data_row(i, x_model) for i in range(len(sent.token_offsets))]
        token_ids = [token_id for data_row in sent_data_rows for token_id in data_row['xfmr_token']]
        if len(token_ids) > x_model.max_seq_len:
            continue
        for data_row in sent_data_rows:
            for k in data_row:
                sent_data[k].extend(data_row[k])
    return pd.DataFrame(sent_data)


def process_char_labeled_sentences(labeled_sentences: list, cx_model: CharXfmrNerModel) -> pd.DataFrame:
    sent_data = defaultdict(list)
    for sent in labeled_sentences:
        sent.token_offsets.insert(0, (-1, 0))
        sent.token_offsets.append((len(sent.text) - 1, len(sent.text)))
        sent_data_rows = [sent.xfmr_data_row(i, cx_model) for i in range(len(sent.token_offsets))]
        for data_row in sent_data_rows:
            for k in data_row:
                sent_data[k].extend(data_row[k])
    return pd.DataFrame(sent_data)


def process_rex_labeled_sentences(labeled_sentences: list, x_model: XfmrNerModel) -> pd.DataFrame:
    sent_data = defaultdict(list)
    for sent in labeled_sentences:
        sent_data_rows = [sent.rex_data_row(i, x_model) for i in range(len(sent.tokens))]
        for data_row in sent_data_rows:
            for k in data_row:
                sent_data[k].extend(data_row[k])
    return pd.DataFrame(sent_data)

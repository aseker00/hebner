import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd
from src.modeling.modeling_token_xfmr import XfmrNerModel
from src.modeling.modeling_char_xfmr import CharXfmrNerModel


spmrl_norm_labels_gpe_org = {'PER': 'PER', 'LOC': 'LOC', 'ORG': 'ORG', 'GPE': 'ORG', 'EVE': 'O', 'ANG': 'O', 'DUC': 'ORG', 'WOA': 'O', 'FAC': 'LOC'}
spmrl_norm_labels_gpe_loc = {'PER': 'PER', 'LOC': 'LOC', 'ORG': 'ORG', 'GPE': 'LOC', 'EVE': 'O', 'ANG': 'O', 'DUC': 'ORG', 'WOA': 'O', 'FAC': 'LOC'}
# norm_labels_gpe_org = {'PER': 'PER', 'LOC': 'LOC', 'ORG': 'ORG', 'GPE': 'ORG', 'EVE': 'ORG', 'ANG': 'ORG', 'DUC': 'ORG', 'WOA': 'ORG', 'FAC': 'ORG'}
# norm_labels_gpe_loc = {'PER': 'PER', 'LOC': 'LOC', 'ORG': 'ORG', 'GPE': 'LOC', 'EVE': 'ORG', 'ANG': 'ORG', 'DUC': 'ORG', 'WOA': 'ORG', 'FAC': 'ORG'}

# PER - person
# ORG - organization
# LOC - location
# GPE - geo-political
# EVE - event
# DUC - product
# FAC - artifact
# ANG - language
# WOA - work of art
def normalize_spmrl(label: str, norm_labels: dict) -> str:
    norm_label = norm_labels.get(label[2:], 'O')
    if norm_label == 'O':
        return norm_label
    norm_prefix = 'B' if label[0] == 'B' or label[0] == 'S' else 'I'
    return norm_prefix + '-' + norm_label


# PER - person
# ORG - organization
# LOC - location
# MISC - miscellaneous
# TTL - title
def normalize_project(label: str) -> str:
    if label in ['MISC', 'TTL']:
        return 'O'
    return label


def save_processed_dataset(dataset: pd.DataFrame, dataset_file_path: Path):
    dataset.to_csv(str(dataset_file_path))


def load_processed_dataset(dataset_file_path: Path) -> pd.DataFrame:
    return pd.read_csv(str(dataset_file_path))


def save_model_token_data_samples(dataset_file_path: Path, labeled_sentences: list, token_df: pd.DataFrame, x_model: XfmrNerModel):
    labeled_sentences = {sent.sent_id: sent for sent in labeled_sentences}
    token_gb = token_df.groupby('sent_idx')
    token_groups = {i: token_gb.get_group(i) for i in token_gb.groups}
    max_token_seq_len = max([len(token_groups[i].index) for i in token_groups])
    data_samples = [x_model.to_sample(i, labeled_sentences[i].text, token_groups[i], max_token_seq_len) for i in token_groups]
    with open(str(dataset_file_path), 'wb') as f:
        pickle.dump(data_samples, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_data_samples(dataset_file_path: Path) -> list:
    with open(str(dataset_file_path), 'rb') as f:
        return pickle.load(f)


def save_model_char_data_samples(dataset_file_path: Path, labeled_sentences: list, token_df: pd.DataFrame, char_df: pd.DataFrame, cx_model: CharXfmrNerModel):
    # merged_df = pd.merge(token_df, char_df, on=['sent_idx', 'token_idx'])
    labeled_sentences = {sent.sent_id: sent for sent in labeled_sentences}
    token_gb = token_df.groupby('sent_idx')
    token_groups = {i: token_gb.get_group(i) for i in token_gb.groups}
    char_gb = char_df.groupby('sent_idx')
    char_groups = {i: char_gb.get_group(i) for i in char_gb.groups}
    # max_token_seq_len = max([len(token_groups[i].index) for i in token_groups])
    max_char_seq_len = max([len(char_groups[i].index) for i in char_groups])
    data_samples = [cx_model.to_sample(i, labeled_sentences[i].text, token_groups[i], char_groups[i], max_char_seq_len) for i in token_groups]
    with open(str(dataset_file_path), 'wb') as f:
        pickle.dump(data_samples, file=f, protocol=pickle.HIGHEST_PROTOCOL)


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
        self.token_offsets = [(token_start_offset, token_offsets[token_start_offset]) for token_start_offset in sorted(token_offsets)]
        self.labels = labels

    @property
    def tokens(self) -> list:
        return [self.text[token_offset[0]:token_offset[1]] for token_offset in self.token_offsets]

    def xfmr_data_row(self, token_idx: int, x_model: XfmrNerModel) -> dict:
        token_offset = self.token_offsets[token_idx]
        if token_idx == 0:
            row_token = x_model.cls_token
            row_label = x_model.labels[0]
        elif token_idx == len(self.token_offsets) - 1:
            row_token = x_model.tokenizer.sep_token
            row_label = x_model.labels[0]
        else:
            row_token = self.text[token_offset[0]:token_offset[1]]
            row_label = self.labels[token_idx - 1]
        row_label_id = x_model.label2id[row_label]
        row_xfmr_tokens, row_xfmr_token_ids = x_model.tokenize(row_token)
        return {'sent_idx': [self.sent_id] * len(row_xfmr_tokens),
                'token_idx': [token_idx] * len(row_xfmr_tokens),
                'token': [row_token] * len(row_xfmr_tokens),
                'token_start_offset': [token_offset[0]] * len(row_xfmr_tokens),
                'token_end_offset': [token_offset[1]] * len(row_xfmr_tokens),
                'xfmr_token': row_xfmr_tokens,
                'xfmr_token_id': row_xfmr_token_ids,
                'token_label': [row_label] * len(row_xfmr_tokens),
                'token_label_id': [row_label_id] * len(row_xfmr_tokens)}

    def rex_data_row(self, token_idx: int, x_model: XfmrNerModel) -> dict:
        token_offset = self.token_offsets[token_idx]
        row_token = self.text[token_offset[0]:token_offset[1]]
        row_label = self.labels[token_idx]
        row_label_id = x_model.label2id[row_label]
        return {'sent_idx': [self.sent_id],
                'token_idx': [token_idx],
                'token': [row_token],
                'token_start_offset': [token_offset[0]],
                'token_end_offset': [token_offset[1]],
                'token_label': [row_label],
                'token_label_id': [row_label_id]}

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


class CharLabeledSentence:

    def __init__(self, sent_id: int, text: str, token_offsets: dict, char_labels: list):
        self.sent_id = sent_id
        self.text = text
        self.token_offsets = [(token_start_offset, token_offsets[token_start_offset]) for token_start_offset in sorted(token_offsets)]
        self.labels = char_labels

    @property
    def tokens(self) -> list:
        return [self.text[token_offset[0]:token_offset[1]] for token_offset in self.token_offsets]

    def xfmr_data_row(self, token_idx: int, cx_model: CharXfmrNerModel) -> dict:
        token_offset = self.token_offsets[token_idx]
        if token_idx == 0:
            row_chars = [cx_model.cls_token]
            row_labels = [cx_model.labels[0]]
        elif token_idx == len(self.token_offsets) - 1:
            row_chars = [cx_model.sep_token]
            row_labels = [cx_model.labels[0]]
        else:
            row_chars = list(self.text[token_offset[0]:token_offset[1]])
            row_labels = self.labels[token_offset[0]:token_offset[1]]
        row_label_ids = [cx_model.label2id[label] for label in row_labels]
        row_char_ids = [cx_model.char2id[c] for c in row_chars]
        return {'sent_idx': [self.sent_id] * len(row_chars),
                'token_idx': [token_idx] * len(row_chars),
                'char': row_chars,
                'char_id': row_char_ids,
                'char_label': row_labels,
                'char_label_id': row_label_ids}

    def to_adm(self) -> dict:
        label_offsets = [pos for token_offset in self.token_offsets for pos in list(range(token_offset[0], token_offset[1]))]
        entity_items = []
        cur_label = 'O'
        cur_start_offset, cur_end_offset = 0, 0
        cur_end_offset = 0
        for label, label_offset in zip(self.labels, label_offsets):
            if label[0] == 'B':
                if cur_label != 'O':
                    entity_item = {'type': cur_label[2:], 'mentions': [{'startOffset': int(cur_start_offset), 'endOffset': int(cur_end_offset)}]}
                    entity_items.append(entity_item)
                cur_start_offset, cur_end_offset = label_offset, label_offset
            elif label[0] == 'I':
                cur_end_offset = label_offset
                if cur_label == 'O':
                    cur_start_offset = label_offset
            elif cur_label != 'O':
                entity_item = {'type': cur_label[2:], 'mentions': [{'startOffset': int(cur_start_offset), 'endOffset': int(cur_end_offset)}]}
                entity_items.append(entity_item)
            cur_label = label
        annotations = {'data': self.text, 'attributes': {'entities': {'type': 'list', 'itemType': 'entities', 'items': entity_items}}}
        return annotations


def process_token_labeled_sentences(labeled_sentences: list, x_model: XfmrNerModel) -> pd.DataFrame:
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
        # token_ids = [token_id for data_row in sent_data_rows for token_id in data_row['xfmr_token']]
        # if len(token_ids) > cx_model.max_seq_len:
        #     continue
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


def get_chars_from_processed_data(data_files: list) -> list:
    char_ids = {}
    for data_file in data_files:
        df = load_processed_dataset(data_file)
        df_char_ids = {a[1]: a[0] for a in df[['char', 'char_id']].to_numpy()}
        char_ids.update(df_char_ids)
    return [char_ids[char_id] for char_id in sorted(char_ids)]

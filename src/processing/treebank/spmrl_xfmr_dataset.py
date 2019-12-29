from transformers import XLMTokenizer, XLMModel
from src.processing import processing_conllu as conllu
from src.processing.processing_utils import *


# PER - person
# ORG - organization
# LOC - location
# GPE - geo-political
# EVE - event
# DUC - product
# FAC - artifact
# ANG - language
# WOA - work of art
def normalize(label: str) -> str:
    # if label[2:] in ['EVE', 'DUC', 'ANG', 'WOA']:
    if label[2:] in ['EVE', 'ANG']:
        return 'O'
    if label[2:] == 'GPE':
        return label[:2] + 'LOC'
        # return label[:2] + 'ORG'
    if label[2:] == 'DUC':
        return label[:2] + 'ORG'
    if label[2:] == 'WOA':
        return label[:2] + 'ORG'
    if label[2:] == 'FAC':
        return label[:2] + 'LOC'
    return label


def _extract_label(token_node: list, normalize_func) -> str:
    label = 'O'
    for node in token_node:
        if node['misc']['biose'] != label:
            if label[0] == 'B':
                pass
            elif node['misc']['biose'][0] == 'E':
                label = 'I' + node['misc']['biose'][1:]
            elif node['misc']['biose'][0] == 'S':
                label = 'B' + node['misc']['biose'][1:]
            else:
                label = node['misc']['biose']
    return normalize_func(label)


def _extract_token(token_node: list) -> (str, int):
    node = token_node[0]
    token = node['misc']['token_str']
    # token_idx = int(node['misc']['token_id'])
    return token


def _extract_sent_id(sentence: dict) -> int:
    return int(sentence['id'])


def _extract_text(sentence: dict) -> str:
    return sentence['text']


def _extract_token_offsets(sentence: dict) -> dict:
    token_offsets = {}
    token_nodes = sentence['token_nodes']
    tokens = [_extract_token(token_node) for token_node in token_nodes]
    text = _extract_text(sentence)
    cur_pos = 0
    for token in tokens:
        token_start_offset = text.find(token, cur_pos)
        if token_start_offset < 0:
            token_start_offset = cur_pos if cur_pos == 0 else cur_pos + 1
        token_end_offset = token_start_offset + len(token)
        token_offsets[token_start_offset] = token_end_offset
        cur_pos = token_end_offset
    return token_offsets


def _label_sentence(sentence: dict, normalize_func) -> LabeledSentence:
    sent_id = _extract_sent_id(sentence)
    text = _extract_text(sentence)
    token_nodes = sentence['token_nodes']
    token_offsets = _extract_token_offsets(sentence)
    tokens = [_extract_token(token_node) for token_node in token_nodes]
    labels = [_extract_label(token_node, normalize_func) for token_node in token_nodes]
    return LabeledSentence(sent_id, text, token_offsets, tokens, labels)


def label_sentences(lattice_sentences: list, normalize_func) -> list:
    return [_label_sentence(ann_sent, normalize_func) for ann_sent in lattice_sentences]


def main():
    xlm_tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
    xlm_model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    ner_model = XfmrNerModel('xlm', xlm_tokenizer, xlm_model)
    data_file_path = Path('data/processed/spmrl-xlm.csv')
    lattice_sentences = conllu.read_conllu(Path('data/clean/treebank/spmrl-02.conllu'), 'spmrl')
    labeled_sentences = label_sentences(lattice_sentences, normalize)
    df = process_labeled_sentences(labeled_sentences, ner_model)
    save_processed_dataset(df, data_file_path)
    # save_model_data_samples(labeled_sentences, df, 'train', ner_model)
    save_model_data_samples(labeled_sentences, df, 'spmrl', ner_model)


if __name__ == "__main__":
    main()

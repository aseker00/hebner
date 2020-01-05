from transformers import XLMTokenizer, XLMModel, BertTokenizer, BertModel
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
def normalize(label: str, gpe_label: str) -> str:
    # if label[2:] in ['EVE', 'DUC', 'ANG', 'WOA']:
    if label[2:] in ['EVE', 'ANG']:
        return 'O'
    if label[2:] == 'GPE':
        return label[:2] + gpe_label
        # return label[:2] + 'ORG'
    if label[2:] == 'DUC':
        return label[:2] + 'ORG'
    if label[2:] == 'WOA':
        return label[:2] + 'ORG'
    if label[2:] == 'FAC':
        return label[:2] + 'LOC'
    return label


def extract_sent_id(sentence: dict) -> int:
    return int(sentence['id'])


def extract_text(sentence: dict) -> str:
    return sentence['text']


def extract_token(token_node: list) -> str:
    tokens = [node['misc']['token_str'] for node in token_node]
    assert tokens.count(tokens[0]) == len(tokens)
    return tokens[0]


def extract_token_offsets(sentence: dict) -> dict:
    token_offsets = {}
    token_nodes = sentence['token_nodes']
    tokens = [extract_token(token_node) for token_node in token_nodes]
    text = extract_text(sentence)
    cur_pos = 0
    for token in tokens:
        token_start_offset = text.find(token, cur_pos)
        if token_start_offset < 0:
            token_start_offset = cur_pos if cur_pos == 0 else cur_pos + 1
        token_end_offset = token_start_offset + len(token)
        token_offsets[token_start_offset] = token_end_offset
        cur_pos = token_end_offset
    return token_offsets


def extract_label(token_node: list) -> str:
    label = 'O'
    for node in token_node:
        if node['misc']['biose'] != label:
            if node['misc']['biose'][0] == 'E':
                label = 'I' + node['misc']['biose'][1:]
            elif node['misc']['biose'][0] == 'S':
                label = 'B' + node['misc']['biose'][1:]
            elif label[0] != 'B':
                label = node['misc']['biose']
    return label


def label_sentence(sentence: dict, gpe_label: str) -> TokenLabeledSentence:
    sent_id = extract_sent_id(sentence)
    text = extract_text(sentence)
    token_nodes = sentence['token_nodes']
    token_offsets = extract_token_offsets(sentence)
    # tokens = [extract_token(token_node) for token_node in token_nodes]
    labels = [normalize(extract_label(token_node), gpe_label) for token_node in token_nodes]
    return TokenLabeledSentence(sent_id, text, token_offsets, labels)


def label_sentences(lattice_sentences: list, gpe_label: str = 'LOC') -> list:
    return [label_sentence(ann_sent, gpe_label) for ann_sent in lattice_sentences]


def main(model_type: str = 'xlm'):
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    ner_model = XfmrNerModel(model_type, tokenizer, model)
    lattice_sentences = conllu.read_conllu(Path('data/clean/treebank/spmrl-07.conllu'), 'spmrl')
    labeled_sentences = label_sentences(lattice_sentences, 'LOC')
    df = process_xfmr_labeled_sentences(labeled_sentences, ner_model)
    data_file_path = Path('data/processed/{}-{}.csv'.format('spmrl', model_type))
    save_processed_dataset(df, data_file_path)
    save_model_data_samples('.', labeled_sentences, df, 'spmrl', ner_model)


if __name__ == "__main__":
    main()

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
def _normalize(label: str) -> str:
    # if label[2:] in ['EVE', 'DUC', 'ANG', 'WOA']:
    if label[2:] in ['EVE', 'ANG']:
        return 'O'
    if label[2:] == 'GPE':
        return label[:2] + 'ORG'
    if label[2:] == 'DUC':
        return label[:2] + 'ORG'
    if label[2:] == 'WOA':
        return label[:2] + 'ORG'
    if label[2:] == 'FAC':
        return label[:2] + 'LOC'
    return label


def _extract_label(token_node: list) -> str:
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
    return label


def _extract_token(token_node: list) -> (str, int):
    node = token_node[0]
    token = node['misc']['token_str']
    token_idx = int(node['misc']['token_id'])
    return token, token_idx


def _get_token_labels(sentence: dict, normalize_func) -> (list, list):
    tokens = []
    labels = []
    token_nodes = sentence['token_nodes']
    for token_node in token_nodes:
        token, token_idx = _extract_token(token_node)
        label = _extract_label(token_node)
        label = normalize_func(label)
        tokens.append(token)
        labels.append(label)
    return tokens, labels


def _label_sentences(sentences: list, normalize_func) -> list:
    labeled_sentences = []
    for i, sentence in enumerate(sentences):
        tokens, labels = _get_token_labels(sentence, normalize_func)
        labeled_sentence = LabeledSentence(i + 1, tokens, labels)
        labeled_sentences.append(labeled_sentence)
    return labeled_sentences


xlm_tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
xlm_model = XLMModel.from_pretrained('xlm-mlm-100-1280')
ner_model = XfmrNerModel(xlm_tokenizer, xlm_model)
data_file_path = Path('data/processed/spmrl-xfmr.csv')
sentences = conllu.read_conllu(Path('data/clean/treebank/spmrl-02.conllu'), 'spmrl')
labeled_sentences = _label_sentences(sentences, _normalize)
df = process_labeled_sentences(labeled_sentences, ner_model)
save_processed_dataset(df, data_file_path)
save_model_data_samples(df, 'train', ner_model)

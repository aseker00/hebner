from transformers import XLMTokenizer, XLMModel
from src.processing import processing_adm as adm
from src.processing.processing_utils import *


# PER - person
# ORG - organization
# LOC - location
# MISC - miscellaneous
# TTL - title
def _normalize(label: str, token: str) -> str:
    if is_english(token):
        return 'O'
    if label in ['MISC', 'TTL']:
        return 'O'
    return label


# PERSON - person
# ORGANIZATION - organization
# LOCATION - location
def _normalize_rex(label: str, token: str) -> str:
    if is_english(token):
        return 'O'
    if label[:3] not in ['PER', 'LOC', 'ORG']:
        # print(label)
        return 'O'
    return label[:3]


def _extract_token(tokens: dict, token_start_offset: int, text: str) -> str:
    token_end_offset = tokens[token_start_offset]
    return text[token_start_offset:token_end_offset]


def _extract_label(labels: dict, token_start_offset: int) -> str:
    return labels[token_start_offset]


def _get_token_labels(sentence: tuple, normalize_func) -> (list, list):
    (text, token_offsets, tag_offsets, entity_offsets,
     entity_type_offsets, entity_token_offsets, token_entity_offsets) = sentence
    label_offsets = {}
    for token_start_offset in sorted(token_offsets):
        if token_start_offset in label_offsets:
            continue
        label = 'O'
        token = _extract_token(token_offsets, token_start_offset, text)
        if token_start_offset in token_entity_offsets:
            entity_start_offset = token_entity_offsets[token_start_offset]
            entity_type = entity_type_offsets[entity_start_offset]
            label = normalize_func(entity_type, token)
            if label != 'O':
                label = '{}-{}'.format('B', label)
            label_offsets[token_start_offset] = label
            if label != 'O':
                label = 'I' + label[1:]
            for in_token_start_offset in entity_token_offsets[entity_start_offset][1:]:
                label_offsets[in_token_start_offset] = label
        else:
            label_offsets[token_start_offset] = label
    tokens = []
    labels = []
    for token_start_offset in sorted(token_offsets):
        token_end_offset = token_offsets[token_start_offset]
        token = text[token_start_offset:token_end_offset]
        label = label_offsets[token_start_offset]
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
for project_type in ['fin', 'news']:
    data_file_path = Path('data/processed/{}-xfmr.csv'.format(project_type))
    sentences = adm.read_project(Path('data/clean/project/{}'.format(project_type)))
    labeled_sentences = _label_sentences(sentences, _normalize)
    df = process_labeled_sentences(labeled_sentences, ner_model)
    save_processed_dataset(df, data_file_path)
    save_model_data_samples(df, 'test' if project_type == 'fin' else 'valid', ner_model)

    rex_data_file_path = Path('data/processed/{}-rex.csv'.format(project_type))
    rex_sentences = adm.read_project(Path('data/clean/project-rex/{}'.format(project_type)))
    rex_labeled_sentences = _label_sentences(rex_sentences, _normalize_rex)
    df = process_rex_labeled_sentences(rex_labeled_sentences, ner_model)
    save_processed_dataset(df, rex_data_file_path)

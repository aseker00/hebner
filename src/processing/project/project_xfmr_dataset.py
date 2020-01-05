from transformers import XLMTokenizer, XLMModel, BertTokenizer, BertModel
from src.processing import processing_adm as adm
from src.processing.processing_utils import *


# PER - person
# ORG - organization
# LOC - location
# MISC - miscellaneous
# TTL - title
def normalize(label: str, token: str) -> str:
    # if is_english(token):
    #     return 'O'
    if label in ['MISC', 'TTL']:
        return 'O'
    return label


# PERSON - person
# ORGANIZATION - organization
# LOCATION - location
def normalize_rex(label: str, token: str) -> str:
    # if is_english(token):
    #     return 'O'
    if label[:3] not in ['PER', 'LOC', 'ORG']:
        # print(label)
        return 'O'
    return label[:3]


def extract_token(tokens: dict, token_start_offset: int, text: str) -> str:
    token_end_offset = tokens[token_start_offset]
    return text[token_start_offset:token_end_offset]


def extract_label(labels: dict, token_start_offset: int) -> str:
    return labels[token_start_offset]


def extract_label_offsets(sentence: tuple, normalize_func):
    text, token_offsets, tag_offsets, entity_offsets, entity_type_offsets, entity_token_offsets, token_entity_offsets = sentence
    label_offsets = {}
    for token_start_offset in sorted(token_offsets):
        if token_start_offset in label_offsets:
            continue
        label = 'O'
        token = extract_token(token_offsets, token_start_offset, text)
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
    return label_offsets


def label_sentence(sent_id :int, sentence: tuple, normalize_func) -> TokenLabeledSentence:
    text, token_offsets, tag_offsets, entity_offsets, entity_type_offsets, entity_token_offsets, token_entity_offsets = sentence
    label_offsets = extract_label_offsets(sentence, normalize_func)
    # tokens = [extract_token(token_offsets, token_start_offset, text) for token_start_offset in sorted(token_offsets)]
    labels = [extract_label(label_offsets, token_start_offset) for token_start_offset in sorted(token_offsets)]
    return TokenLabeledSentence(sent_id, text, token_offsets, labels)


def label_sentences(annotated_sentences: list, normalize_func) -> list:
    return [label_sentence(i + 1, ann_sent, normalize_func) for i, ann_sent in enumerate(annotated_sentences)]


def main(model_type: str = 'xlm'):
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    ner_model = XfmrNerModel(model_type, tokenizer, model)
    for project_type in ['fin', 'news']:
        data_file_path = Path('data/processed/{}-{}.csv'.format(project_type, model_type))
        annotated_sentences = adm.read_project(Path('data/clean/project/{}'.format(project_type)))
        labeled_sentences = label_sentences(annotated_sentences, normalize)
        df = process_xfmr_labeled_sentences(labeled_sentences, ner_model)
        save_processed_dataset(df, data_file_path)
        save_model_data_samples('.', labeled_sentences, df, project_type, ner_model)


if __name__ == "__main__":
    main()

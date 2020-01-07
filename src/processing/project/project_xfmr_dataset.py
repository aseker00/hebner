from pathlib import Path
from transformers import XLMTokenizer, XLMModel, BertTokenizer, BertModel
from src.modeling.modeling_xfmr import XfmrNerModel
from src.processing import processing_adm as adm
from src.processing.processing_utils import process_xfmr_labeled_sentences, TokenLabeledSentence
from src.processing.processing_utils import save_processed_dataset, save_model_data_samples


# PER - person
# ORG - organization
# LOC - location
# MISC - miscellaneous
# TTL - title
def normalize(label: str) -> str:
    if label in ['MISC', 'TTL']:
        return 'O'
    return label


# PERSON - person
# ORGANIZATION - organization
# LOCATION - location
def normalize_rex(label: str) -> str:
    if label[:3] not in ['PER', 'LOC', 'ORG']:
        return 'O'
    return label[:3]


def extract_token_label_offsets(sentence: tuple, normalize_func):
    _, token_offsets, _, _, entity_type_offsets, entity_token_offsets, token_entity_offsets = sentence
    label_offsets = {}
    for token_start_offset in sorted(token_offsets):
        if token_start_offset in label_offsets:
            continue
        label = 'O'
        if token_start_offset in token_entity_offsets:
            entity_start_offset = token_entity_offsets[token_start_offset]
            entity_type = normalize_func(entity_type_offsets[entity_start_offset])
            label = entity_type if entity_type == 'O' else 'B-' + entity_type
            label_offsets[token_start_offset] = label
            label = entity_type if entity_type == 'O' else 'I-' + entity_type
            for in_token_start_offset in entity_token_offsets[entity_start_offset][1:]:
                label_offsets[in_token_start_offset] = label
        else:
            label_offsets[token_start_offset] = label
    return label_offsets


def label_token_sentence(sent_id :int, sentence: tuple, normalize_func) -> TokenLabeledSentence:
    text, token_offsets, _, _, _, _, _ = sentence
    token_label_offsets = extract_token_label_offsets(sentence, normalize_func)
    token_labels = [token_label_offsets[token_start_offset] for token_start_offset in sorted(token_offsets)]
    return TokenLabeledSentence(sent_id, text, token_offsets, token_labels)


def label_token_sentences(annotated_sentences: list, normalize_func) -> list:
    return [label_token_sentence(i + 1, ann_sent, normalize_func) for i, ann_sent in enumerate(annotated_sentences)]


def main(model_type: str = 'xlm'):
    project_sentences = {}
    for project_type in ['news', 'fin']:
        annotated_sentences = adm.read_project(Path('data/clean/project/{}'.format(project_type)))
        token_labeled_sentences = label_token_sentences(annotated_sentences, normalize)
        project_sentences[project_type] = token_labeled_sentences

    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    ner_model = XfmrNerModel(model_type, tokenizer, model)

    for project_type in ['news', 'fin']:
        token_labeled_sentences = project_sentences[project_type]
        df = process_xfmr_labeled_sentences(token_labeled_sentences, ner_model)
        dataset_name = '{}-{}'.format(project_type, model_type)
        data_file_path = Path('data/processed/{}.csv'.format(dataset_name))
        save_processed_dataset(df, data_file_path)
        sample_file_path = Path('data/processed/{}.pkl'.format(dataset_name))
        save_model_data_samples(sample_file_path, token_labeled_sentences, df, ner_model)


if __name__ == "__main__":
    main()

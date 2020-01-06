from pathlib import Path
import fasttext
import pandas as pd
from transformers import XLMTokenizer, XLMModel, BertTokenizer, BertModel
from src.modeling.modeling_char_xfmr import CharXfmrNerModel
from src.modeling.modeling_xfmr import XfmrNerModel
from src.processing import processing_adm as adm
from src.processing.processing_utils import CharLabeledSentence, process_char_labeled_sentences
from src.processing.processing_utils import save_processed_dataset, load_processed_dataset, save_char_model_data_samples


def normalize(label: str) -> str:
    if label in ['MISC', 'TTL']:
        return 'O'
    return label


norm_rex_labels = {'PERSON': 'PER', 'LOCATION': 'LOC', 'ORGANIZATION': 'ORG'}


def normalize_rex(label: str) -> str:
    if label[:3] not in ['PER', 'LOC', 'ORG']:
        return 'O'
    return label[:3]


def label_char_sentence(sent_id :int, sentence: tuple, normalize_func) -> CharLabeledSentence:
    text, token_offsets, _, entity_offsets, entity_type_offsets, _, _ = sentence
    char_labels = ['O'] * len(text)
    for entity_start_offset in entity_offsets:
        entity_type = normalize_func(entity_type_offsets[entity_start_offset])
        if entity_type == 'O':
            continue
        label = 'B-' + entity_type
        char_labels[entity_start_offset] = label
        label = 'I-' + entity_type
        entity_end_offset = entity_offsets[entity_start_offset]
        char_labels[entity_start_offset + 1:entity_end_offset] = [label] * (entity_end_offset - entity_start_offset - 1)
    return CharLabeledSentence(sent_id, text, token_offsets, char_labels)


def label_char_sentences(annotated_sentences: list, normalize_func) -> list:
    return [label_char_sentence(i + 1, ann_sent, normalize_func) for i, ann_sent in enumerate(annotated_sentences)]


def main(model_type: str = 'xlm'):
    project_sentences = {}
    for project_type in ['news', 'fin']:
        annotated_sentences = adm.read_project(Path('data/clean/project/{}'.format(project_type)))
        char_labeled_sentences = label_char_sentences(annotated_sentences, normalize)
        project_sentences[project_type] = char_labeled_sentences
    chars = set()
    for project_type in project_sentences:
        char_labeled_sentences = project_sentences[project_type]
        sent_tokens = [sent.text[token_offset[0]:token_offset[1]] for sent in char_labeled_sentences for token_offset in
                       sent.token_offsets]
        sent_chars = sorted(list(set([c for token in sent_tokens for c in list(token)])))
        chars.update(sent_chars)
    char_data_file_path = Path('data/processed/{}-{}-{}.csv'.format('spmrl', model_type, 'char'))
    train_df = load_processed_dataset(char_data_file_path)
    char2id = {a[0]: a[1] for a in train_df[['char', 'char_id']].to_numpy()}
    diff_chars = sorted(list(chars.difference(char2id.keys())))
    for c in diff_chars:
        char2id[c] = len(char2id) + 1
    ft_model = fasttext.load_model("model/ft/cc.he.300.bin")
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    x_model = XfmrNerModel(model_type, tokenizer, model)
    char2id[x_model.pad_token] = 0
    char2id = {k: v for k, v in sorted(char2id.items(), key=lambda item: item[1])}
    ner_model = CharXfmrNerModel(x_model, ft_model, char2id)

    for project_type in ['news', 'fin']:
        char_labeled_sentences = project_sentences[project_type]
        char_df = process_char_labeled_sentences(char_labeled_sentences, ner_model)
        token_data_file_path = Path('data/processed/{}-{}.csv'.format(project_type, model_type))
        token_df = load_processed_dataset(token_data_file_path)
        m = char_df.sent_idx.isin(token_df.sent_idx)
        char_df = char_df[m]
        # merged_df = pd.merge(token_df, char_df, on=['sent_idx', 'token_idx'])
        data_file_path = Path('data/processed/{}-{}-{}.csv'.format(project_type, model_type, 'char'))
        save_processed_dataset(char_df, data_file_path)
        save_char_model_data_samples('.', char_labeled_sentences, token_df, char_df, project_type, ner_model)


if __name__ == "__main__":
    main()

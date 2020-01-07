from pathlib import Path
import fasttext
from transformers import BertTokenizer, BertModel, XLMTokenizer, XLMModel
from src.modeling.modeling_xfmr import XfmrNerModel
from src.modeling.modeling_char_xfmr import CharXfmrNerModel
from src.processing.processing_utils import CharLabeledSentence
from src.processing.processing_utils import save_processed_dataset, load_processed_dataset
from src.processing.processing_utils import process_char_labeled_sentences, save_char_model_data_samples
from src.processing.treebank.spmrl_xfmr_dataset import extract_token
from src.processing import processing_conllu as conllu


# norm_labels_gpe_org = {'PER': 'PER', 'LOC': 'LOC', 'ORG': 'ORG', 'GPE': 'ORG', 'EVE': 'ORG', 'ANG': 'ORG', 'DUC': 'ORG', 'WOA': 'ORG', 'FAC': 'ORG'}
# norm_labels_gpe_loc = {'PER': 'PER', 'LOC': 'LOC', 'ORG': 'ORG', 'GPE': 'LOC', 'EVE': 'ORG', 'ANG': 'ORG', 'DUC': 'ORG', 'WOA': 'ORG', 'FAC': 'ORG'}
norm_labels_gpe_org = {'PER': 'PER', 'LOC': 'LOC', 'ORG': 'ORG', 'GPE': 'ORG', 'EVE': 'O', 'ANG': 'O', 'DUC': 'ORG', 'WOA': 'O', 'FAC': 'LOC'}
norm_labels_gpe_loc = {'PER': 'PER', 'LOC': 'LOC', 'ORG': 'ORG', 'GPE': 'LOC', 'EVE': 'O', 'ANG': 'O', 'DUC': 'ORG', 'WOA': 'O', 'FAC': 'LOC'}


def normalize(label: str, norm_labels: dict) -> str:
    norm_label = norm_labels.get(label[2:], 'O')
    if norm_label == 'O':
        return norm_label
    norm_prefix = 'B' if label[0] == 'B' or label[0] == 'S' else 'I'
    return norm_prefix + '-' + norm_label


def extract_tokenized_label(token_node: list) -> str:
    labels = [node['misc']['biose'] for node in token_node]
    if len(labels) > 1:
        try:
            labels.remove('O')
        except ValueError:
            return labels[0]
    return labels[0]


def extract_segmented_labels(token_node: list) -> list:
    return [node['misc']['biose'] for node in token_node]


def get_sent_id(sentence: dict) -> int:
    return int(sentence['id'])


def get_text(sentence: dict) -> str:
    return sentence['text']


def extract_token_offsets(sentence: dict) -> list:
    offsets = []
    start_offset, end_offset = 0, 0
    token_nodes = sentence['token_nodes']
    text = get_text(sentence)
    for token_node in token_nodes:
        token = extract_token(token_node)
        assert text[start_offset:start_offset + len(token)] == token
        end_offset = start_offset + len(token)
        offsets.append((start_offset, end_offset, token))
        start_offset = end_offset
        space_after = bool(token_node[-1]['misc']['SpaceAfter'])
        if space_after:
            start_offset += 1
    return offsets


suffix_chars_map = {'ן': 'נ', 'ם': 'מ', 'ך': 'כ', 'ץ': 'צ', 'ף': 'פ'}


def extract_segment_offsets(sentence: dict) -> list:
    offsets = []
    token_nodes = sentence['token_nodes']
    text = get_text(sentence)
    sent_id = get_sent_id(sentence)
    token_offsets = extract_token_offsets(sentence)
    for (token_start_offset, token_end_offset, token), token_node in zip(token_offsets, token_nodes):
        start_offset, end_offset = token_start_offset, token_end_offset
        for i in range(len(token_node)):
            node = token_node[i]
            seg = node['form']
            tag = node['postag']
            if tag == 'IN' and seg == 'עם' and text[start_offset:start_offset+len(seg)+1] == 'עימ':
                seg = 'עימ'
            elif seg == 'לפתור' and text[start_offset:start_offset+len(seg)] == 'לפותר':
                seg = 'לפותר'
            elif seg == 'לחקור' and text[start_offset:start_offset+len(seg)] == 'לחוקר':
                seg = 'לחוקר'
            if ((seg == text[start_offset:start_offset + len(seg)]) or
                    (seg[:-1] + suffix_chars_map.get(seg[-1], seg[-1]) == text[start_offset:start_offset + len(seg)])):
                if tag == 'DEF' and 0 < i < len(token_node) - 1:
                    prev_seg = token_node[i - 1]['form']
                    prev_tag = token_node[i - 1]['postag']
                    if prev_tag == 'PREPOSITION' and prev_seg == 'ב' or prev_seg == 'ל':
                        next_seg = token_node[i + 1]['form']
                        if next_seg[0] == 'ה':
                            start_offset = start_offset
                            end_offset = end_offset
                        else:
                            end_offset = start_offset + len(seg)
                    else:
                        end_offset = start_offset + len(seg)
                else:
                    end_offset = start_offset + len(seg)
            elif tag == 'DEF' and 0 < i < len(token_node) - 1:
                start_offset = start_offset
                end_offset = end_offset
            elif tag == 'DUMMY_AT' and 0 < i < len(token_node) - 1:
                start_offset = start_offset
                end_offset = end_offset
            elif tag == 'POS' and seg == 'אות' and 0 < i < len(token_node) - 1:
                start_offset = start_offset
                end_offset = end_offset
            elif tag == 'AT':
                start_offset = start_offset
                end_offset = end_offset
            elif tag == 'S_PRN':
                end_offset = token_end_offset
            elif tag == 'S_ANP':
                end_offset = token_end_offset
            elif tag == 'PRP' and i == len(token_node) - 1:
                end_offset = token_end_offset
            elif seg == 'הכל' and i == len(token_node) - 1:
                end_offset = token_end_offset
            elif tag == 'IN' and seg == 'עם' and i == 0 and len(token_node) > 1:
                start_offset = start_offset
                end_offset = end_offset
            else:
                print('sent_id {}: {}_{}'.format(sent_id, seg, tag))
            offsets.append((start_offset, end_offset, seg))
            start_offset = end_offset
    return offsets


def extract_tokenized_label_offsets(sentence: dict, norm_labels: dict) -> list:
    offsets = []
    token_nodes = sentence['token_nodes']
    token_labels = [extract_tokenized_label(token_node) for token_node in token_nodes]
    token_labels = [normalize(label, norm_labels) for label in token_labels]
    token_offsets = extract_token_offsets(sentence)
    for token_label, (start_offset, end_offset, token) in zip(token_labels, sorted(token_offsets)):
        offsets.append((start_offset, end_offset, token, token_label))
    return offsets


def extract_segmented_label_offsets(sentence: dict, norm_labels: dict) -> list:
    offsets = []
    token_nodes = sentence['token_nodes']
    seg_labels = [seg_label for token_node in token_nodes for seg_label in extract_segmented_labels(token_node)]
    seg_labels = [normalize(label, norm_labels) for label in seg_labels]
    seg_offsets = extract_segment_offsets(sentence)
    for seg_label, (start_offset, end_offset, seg) in zip(seg_labels, seg_offsets):
        offsets.append((start_offset, end_offset, seg, seg_label))
    return offsets


def label_char_sentence(sentence: dict, norm_labels: dict) -> CharLabeledSentence:
    sent_id = get_sent_id(sentence)
    text = get_text(sentence)
    tokenized_label_offsets = extract_tokenized_label_offsets(sentence, norm_labels)
    token_offsets = {offset[0]:offset[1] for offset in tokenized_label_offsets}
    segmented_label_offsets = extract_segmented_label_offsets(sentence, norm_labels)
    seg_offsets = {}
    seg_labels = {}
    for seg_label_offset in segmented_label_offsets:
        seg_start_offset = seg_label_offset[0]
        seg_end_offset = seg_label_offset[1]
        seg_label = seg_label_offset[-1]
        if seg_start_offset in seg_labels:
            seg_label = seg_labels[seg_start_offset]
        seg_offsets[seg_start_offset] = seg_end_offset
        seg_labels[seg_start_offset] = seg_label
    char_labels = ['O'] * len(text)
    for seg_start_offset in seg_offsets:
        seg_label = seg_labels[seg_start_offset]
        if seg_label == 'O':
            continue
        char_labels[seg_start_offset] = seg_label
        if seg_label[0] == 'B':
            seg_label = 'I' + seg_label[1:]
        else:
            char_labels[seg_end_offset:seg_start_offset] = [seg_label] * (seg_start_offset - seg_end_offset)
        seg_end_offset = seg_offsets[seg_start_offset]
        char_labels[seg_start_offset+1:seg_end_offset] = [seg_label] * (seg_end_offset - seg_start_offset - 1)
    return CharLabeledSentence(sent_id, text, token_offsets, char_labels)


def label_char_sentences(lattice_sentences: list, norm_labels: dict) -> list:
    return [label_char_sentence(ann_sent, norm_labels) for ann_sent in lattice_sentences]


def main(model_type: str = 'xlm'):
    gpe_label = 'ORG'
    lattice_sentences = conllu.read_conllu_sentences(Path('data/clean/treebank/spmrl-07.conllu'), 'spmrl')
    norm_labels = norm_labels_gpe_loc if gpe_label == 'LOC' else norm_labels_gpe_org
    char_labeled_sentences = label_char_sentences(lattice_sentences, norm_labels)
    sent_tokens = [sent.text[token_offset[0]:token_offset[1]] for sent in char_labeled_sentences for token_offset in
                   sent.token_offsets]
    sent_chars = sorted(list(set([c for token in sent_tokens for c in list(token)])))
    ft_model = fasttext.load_model("model/ft/cc.he.300.bin")
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    x_model = XfmrNerModel(model_type, tokenizer, model)
    tokenizer_chars = [x_model.pad_token] + sorted(list({x_model.cls_token, x_model.sep_token}))
    chars = tokenizer_chars + sent_chars
    char2id = {c: i for i, c in enumerate(chars)}
    char2id = {k: v for k, v in sorted(char2id.items(), key=lambda item: item[1])}
    ner_model = CharXfmrNerModel(x_model, ft_model, char2id)
    char_df = process_char_labeled_sentences(char_labeled_sentences, ner_model)
    dataset_name = '{}-{}-{}-{}'.format('spmrl', 'char', model_type, 'gpe-loc' if gpe_label == 'LOC' else 'gpe-org')
    data_file_path = Path('data/processed/{}.csv'.format(dataset_name))
    save_processed_dataset(char_df, data_file_path)
    token_dataset_name = '{}-{}-{}'.format('spmrl', model_type, 'gpe-loc' if gpe_label == 'LOC' else 'gpe-org')
    token_data_file_path = Path('data/processed/{}.csv'.format(token_dataset_name))
    token_df = load_processed_dataset(token_data_file_path)
    sample_file_path = Path('data/processed/{}.pkl'.format(dataset_name))
    save_char_model_data_samples(sample_file_path, char_labeled_sentences, token_df, char_df, ner_model)


if __name__ == "__main__":
    main()

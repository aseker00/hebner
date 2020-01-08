from src.processing.processing_utils import CharLabeledSentence, normalize_spmrl
from src.processing.treebank.spmrl_token_xfmr_dataset import extract_token, extract_token_label


def extract_segment_labels(token_node: list) -> list:
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
    token_labels = [extract_token_label(token_node) for token_node in token_nodes]
    token_labels = [normalize_spmrl(label, norm_labels) for label in token_labels]
    token_offsets = extract_token_offsets(sentence)
    for token_label, (start_offset, end_offset, token) in zip(token_labels, sorted(token_offsets)):
        offsets.append((start_offset, end_offset, token, token_label))
    return offsets


def extract_segmented_label_offsets(sentence: dict, norm_labels: dict) -> list:
    offsets = []
    token_nodes = sentence['token_nodes']
    seg_labels = [seg_label for token_node in token_nodes for seg_label in extract_segment_labels(token_node)]
    seg_labels = [normalize_spmrl(label, norm_labels) for label in seg_labels]
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

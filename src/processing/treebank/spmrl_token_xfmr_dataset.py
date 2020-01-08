from src.processing.processing_utils import TokenLabeledSentence, normalize_spmrl


# # PER - person
# # ORG - organization
# # LOC - location
# # GPE - geo-political
# # EVE - event
# # DUC - product
# # FAC - artifact
# # ANG - language
# # WOA - work of art
# def normalize(label: str, gpe_label: str) -> str:
#     # if label[2:] in ['EVE', 'DUC', 'ANG', 'WOA']:
#     if label[2:] in ['EVE', 'ANG']:
#         return 'O'
#     if label[2:] == 'GPE':
#         return label[:2] + gpe_label
#         # return label[:2] + 'ORG'
#     if label[2:] == 'DUC':
#         return label[:2] + 'ORG'
#     if label[2:] == 'WOA':
#         return label[:2] + 'ORG'
#     if label[2:] == 'FAC':
#         return label[:2] + 'LOC'
#     return label


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


def extract_token_label(token_node: list) -> str:
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


def label_token_sentence(sentence: dict, norm_labels: dict) -> TokenLabeledSentence:
    sent_id = extract_sent_id(sentence)
    text = extract_text(sentence)
    token_nodes = sentence['token_nodes']
    token_offsets = extract_token_offsets(sentence)
    labels = [normalize_spmrl(extract_token_label(token_node), norm_labels) for token_node in token_nodes]
    return TokenLabeledSentence(sent_id, text, token_offsets, labels)


def label_token_sentences(lattice_sentences: list, norm_labels: dict) -> list:
    return [label_token_sentence(ann_sent, norm_labels) for ann_sent in lattice_sentences]

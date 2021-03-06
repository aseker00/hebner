from src.processing.processing_utils import TokenLabeledSentence, normalize_spmrl


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
    labels = [node['misc']['biose'] for node in token_node]
    if len(labels) > 1:
        labels = [label for label in labels if label != 'O']
        if not labels:
            return 'O'
        # try:
        #     labels.remove('O')
        # except ValueError:
        #     return labels[0]
    return labels[0]


def label_token_sentence(sentence: dict, norm_labels: dict) -> TokenLabeledSentence:
    sent_id = extract_sent_id(sentence)
    text = extract_text(sentence)
    token_nodes = sentence['token_nodes']
    token_offsets = extract_token_offsets(sentence)
    token_labels = [extract_token_label(token_node) for token_node in token_nodes]
    token_labels = [normalize_spmrl(label, norm_labels) for label in token_labels]
    # labels = [normalize_spmrl(extract_token_label(token_node), norm_labels) for token_node in token_nodes]
    return TokenLabeledSentence(sent_id, text, token_offsets, token_labels)


def label_token_sentences(lattice_sentences: list, norm_labels: dict) -> list:
    return [label_token_sentence(ann_sent, norm_labels) for ann_sent in lattice_sentences]

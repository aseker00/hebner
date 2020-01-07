from collections import defaultdict
from pathlib import Path
import pandas as pd


def read_file(path: Path) -> list:
    with open(str(path), 'r') as f:
        return f.readlines()


spmrl_comment_fields = {
    'sent_id': 'id',
    'global_sent_id': 'global_id',
    'text_from_ud': 'text',
    'very_similar_sent_id': 'very_similar_ids',
    'duplicate_sent_id': 'duplicate_ids'
}


ud_comment_fields = {
    'sent_id': 'id',
    'global_sent_id': 'global_id',
    'text': 'text',
    'very_similar_sent_id': 'very_similar_ids',
    'duplicate_sent_id': 'duplicate_ids'
}

conllu_fields = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats_str', 'head', 'deprel', 'dummy', 'misc_str']
misc_fields = ['biose', 'ner_escaped', 'token_id', 'token_str', 'SpaceAfter']


def parse_spmrl_comment(sent: dict, line: str):
    pos = line.index(' = ')
    field_name = line[2:pos]
    field_value = line[pos + len(' = '):]
    if field_name == 'sent_id' or field_name == 'global_sent_id':
        sent[spmrl_comment_fields[field_name]] = int(field_value)
    elif field_name == 'very_similar_sent_id' or field_name == 'duplicate_sent_id':
        sent[spmrl_comment_fields[field_name]] = set([int(sent_id[1:-1]) for sent_id in field_value[1:-1].split(', ') if sent_id])
    else:
        sent[spmrl_comment_fields[field_name]] = field_value


def parse_ud_comment(sent: dict, line: str):
    pos = line.index(' = ')
    field_name = line[2:pos]
    field_value = line[pos + len(' = '):]
    if field_name == 'sent_id' or field_name == 'global_sent_id':
        sent[ud_comment_fields[field_name]] = int(field_value)
    elif field_name == 'very_similar_sent_id' or field_name == 'duplicate_sent_id':
        sent[ud_comment_fields[field_name]] = set([int(sent_id) for sent_id in field_value[1:-1].split(', ') if sent_id])
    else:
        sent[ud_comment_fields[field_name]] = field_value


def parse_ud_node(conllu_parts: list) -> dict:
    node = {k: v for k, v in zip(conllu_fields, conllu_parts)}
    node['id'] = int(node['id'])
    node['head'] = int(node['head'])
    feat_parts = node['feats_str'].split('|')
    feats = {part.split('=')[0]: part.split('=')[1] for part in feat_parts if part != '_'}
    node['feats'] = feats
    misc_parts = node['misc_str'].split('|')
    misc = {part.split('=')[0]: part.split('=')[1] for part in misc_parts}
    misc['token_id'] = int(misc['token_id'])
    space_after = 'SpaceAfter' not in misc or misc['SpaceAfter'] != 'No'
    misc['SpaceAfter'] = space_after
    node['misc'] = misc
    return node


def parse_spmrl_node(conllu_parts: list) -> dict:
    node = {k: v for k, v in zip(conllu_fields, conllu_parts)}
    node['id'] = int(node['id'])
    node['head'] = int(node['head'])
    feat_parts = node['feats_str'].split('|')
    feats = {part.split('=')[0]: part.split('=')[1] for part in feat_parts if part != '_'}
    node['feats'] = feats
    misc_parts = node['misc_str'].split('|')
    misc = {part.split('=')[0]: part.split('=')[1] for part in misc_parts}
    misc['token_id'] = int(misc['token_id'])
    node['misc'] = misc
    return node


def parse_spmrl_token_node(conllu_parts: list) -> dict:
    node = {k: v for k, v in zip(conllu_fields, conllu_parts)}
    return {'from_node_id': int(node['id'].split('-')[0]),
            'to_node_id': int(node['id'].split('-')[1]),
            'token_str': node['form']}


def parse_ud_token_node(conllu_parts: list) -> dict:
    node = {k: v for k, v in zip(conllu_fields, conllu_parts)}
    misc_parts = node['misc_str'].split('=')
    misc = {} if len(misc_parts) == 1 else {misc_parts[0]: misc_parts[1]}
    space_after = 'SpaceAfter' not in misc or misc['SpaceAfter'] != 'No'
    misc['SpaceAfter'] = space_after
    return {'from_node_id': int(node['id'].split('-')[0]),
            'to_node_id': int(node['id'].split('-')[1]),
            'token_str': node['form'],
            'misc': misc}


def parse_ud_sentence(lines: list) -> dict:
    nodes = []
    token_nodes = []
    sent = {}
    token_node = None
    for line in lines:
        line = line.strip()
        if line[0] == '#':
            parse_ud_comment(sent, line)
        else:
            conllu_parts = line.split()
            if len(conllu_parts[0].split('-')) > 1:
                token_node = parse_ud_token_node(conllu_parts)
            else:
                node = parse_ud_node(conllu_parts)
                nodes.append(node)
                if not token_node or (node['id'] == token_node['from_node_id']):
                    token_nodes.append([node])
                else:
                    if token_node and node['id'] == token_node['to_node_id']:
                        node['misc']['SpaceAfter'] = token_node['misc']['SpaceAfter']
                        token_node = None
                    token_nodes[-1].append(node)
    sent['nodes'] = nodes
    sent['token_nodes'] = token_nodes
    return sent


def parse_spmrl_sentence(lines: list) -> dict:
    nodes = []
    token_ids = defaultdict(list)
    token_nodes = {}
    sent = {}
    token_node = None
    for line in lines:
        line = line.strip()
        if line[0] == '#':
            parse_spmrl_comment(sent, line)
        else:
            conllu_parts = line.split()
            if len(conllu_parts[0].split('-')) > 1:
                token_node = parse_spmrl_token_node(conllu_parts)
            else:
                node = parse_spmrl_node(conllu_parts)
                nodes.append(node)
                token_id = node['misc']['token_id']
                token_ids[token_id].append(node)
                if token_node:
                    token_nodes[token_id] = token_node
                    token_node = None
    text_index = 0
    for token_id in token_ids:
        for i, node in enumerate(token_ids[token_id][::-1]):
            if i == 0:
                text_index += len(node['misc']['token_str'])
                space_after = text_index < len(sent['text']) and sent['text'][text_index] == ' '
                node['misc']['SpaceAfter'] = space_after
                if space_after:
                    text_index += 1
            else:
                node['misc']['SpaceAfter'] = False
    token_nodes = [token_ids[token_id] for token_id in token_ids]
    sent['nodes'] = nodes
    sent['token_nodes'] = token_nodes
    return sent


def to_dataframe_rows(sentence: dict) -> list:
    rows = []
    for node in sentence['nodes']:
        misc = node.pop('misc')
        node.pop('feats')
        # values = [sent['id'], misc['token_id'], misc['token_str'], misc['biose']] + list(node.values())
        # row = {k: v for k, v in zip(keys, values)}
        node['sent_id'] = sentence['id']
        node['token_id'] = misc['token_id']
        node['token_str'] = misc['token_str']
        node['biose'] = misc['biose']
        node['space_after'] = misc['SpaceAfter']
        rows.append(node)
    return rows


def read_conllu_sentences(path: Path, conll_type: str) -> list:
    file_lines = read_file(path)
    empty_line_nums = [-1] + [num for num, line in enumerate(file_lines) if not line.strip()]
    sent_sep = [(start_line_num + 1, end_line_num) for start_line_num, end_line_num in zip(empty_line_nums[:-1], empty_line_nums[1:])]
    sent_lines = [file_lines[start_line_num:end_line_num] for (start_line_num, end_line_num) in sent_sep]
    if conll_type == 'spmrl':
        sentences = [parse_spmrl_sentence(lines) for lines in sent_lines]
    else:
        sentences = [parse_ud_sentence(lines) for lines in sent_lines]
    return sentences


def read_conllu_dataframe(path: Path, conll_type: str) -> pd.DataFrame:
    sentences = read_conllu_sentences(path, conll_type)
    rows = [row for sentence in sentences for row in to_dataframe_rows(sentence)]
    keys = ['sent_id', 'token_id', 'token_str', 'biose', 'id', 'form', 'lemma', 'cpostag', 'postag', 'feats_str',
            'head', 'deprel', 'dummy', 'misc_str', 'space_after']
    return pd.DataFrame(rows, columns=keys)


def filter_sentences(sentences: list, field_name: str):
    ids_to_filter = set()
    duplicate_ids = []
    for sentence in sentences:
        if sentence[field_name]:
            duplicate_set = sentence[field_name] | {sentence['id']}
            if duplicate_set not in duplicate_ids:
                duplicate_ids.append(duplicate_set)
    for ids in duplicate_ids:
        filter_id = False
        for sent_id in ids:
            if not filter_id:
                filter_id = True
            else:
                ids_to_filter.add(sent_id)
    return ids_to_filter

from copy import deepcopy
from pathlib import Path
from src.processing import processing_conllu as conllu


def get_text(sentence: dict) -> str:
    return sentence['text']


def extract_token(token_node: list) -> str:
    tokens = [node['misc']['token_str'] for node in token_node]
    assert tokens.count(tokens[0]) == len(tokens)
    return tokens[0]


def fix(sentence: dict) -> dict:
    token_nodes = sentence['token_nodes']
    tokens = [extract_token(token_node) for token_node in token_nodes]
    spaces = [bool(token_node[-1]['misc']['SpaceAfter']) for token_node in token_nodes]
    text = []
    for token, space in zip(tokens, spaces):
        text.append(token)
        if space:
            text.append(' ')
    text = ''.join(text)
    fixed_sentence = deepcopy(sentence)
    fixed_sentence['text'] = text
    return fixed_sentence


with open('data/clean/treebank/spmrl-02.conllu') as f:
    lines = f.readlines()
sentences = {}
sent_lines = []
for line in [l.strip() for l in lines]:
    sent_lines.append(line)
    if not line:
        sentences[sent_id] = sent_lines
        sent_lines = []
    elif '# sent_id = ' in line:
        sent_id = int(line[len('# sent_id = '):])

lattice_sentences = conllu.read_conllu_sentences(Path('data/clean/treebank/spmrl-02.conllu'), 'spmrl')
fixed_sentences = [fix(sent) for sent in lattice_sentences]
fixed_sent_texts = {sent['id']: sent['text'] for sent in fixed_sentences}
with open('data/clean/treebank/spmrl-03.conllu', 'w') as f:
    for sent_id in sentences:
        sent_text = fixed_sent_texts[sent_id]
        sent_lines = sentences[sent_id]
        sent_lines[2] = '# text_from_ud = {}'.format(sent_text)
        sent_lines = '\n'.join(sent_lines)
        f.write(sent_lines)
        f.write('\n')

from pathlib import Path
from src.processing import processing_conllu as conllu


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def extract_token(token_node: list) -> str:
    tokens = [node['misc']['token_str'] for node in token_node]
    assert tokens.count(tokens[0]) == len(tokens)
    return tokens[0]


def fix(sentence: dict) -> dict:
    fixed_forms = {}
    sent_id = sentence['id']
    token_nodes = sentence['token_nodes']
    for token_node in token_nodes:
        token = extract_token(token_node)
        for i, node in enumerate(token_node):
            seg = node['form']
            tag = node['postag']
            node_id = node['id']
            if tag == 'AT' and seg == 'את':
                if i == 0 and len(token_node) == 2 and token_node[1]['postag'] == 'S_PRN':
                    if token[:3] == 'אות':
                        print('{}, {} -> {}'.format(token, seg, 'אות'))
                        fixed_forms[(sent_id, node_id)] = 'אות'
                # if len(token_node) - 1 > i and (token_node[i + 1]['postag'] == 'S_PRN' or token_node[i + 1]['postag'] == 'PRP'):
                #     if i > 0 or token[:2] != 'את':
                #         print('{}, {} -> {}'.format(token, seg, 'אות'))
                #         fixed_forms[(sent_id, node_id)] = 'אות'
    return fixed_forms


with open('data/clean/treebank/spmrl-06.conllu') as f:
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

lattice_sentences = conllu.read_conllu(Path('data/clean/treebank/spmrl-06.conllu'), 'spmrl')
fixed_forms = [fix(sent) for sent in lattice_sentences]
fixed_forms = {k: v for fixed_form in fixed_forms for k, v in fixed_form.items()}
with open('data/clean/treebank/spmrl-07.conllu', 'w') as f:
    for sent_id in sentences:
        sent_lines = sentences[sent_id]
        fixed_sent_lines = []
        for line in sent_lines:
            if not line:
                fixed_sent_lines.append(line)
            else:
                line_parts = line.split()
                line_node_id = line_parts[0]
                if is_number(line_node_id):
                    node_id = int(line_node_id)
                    if (sent_id, node_id) in fixed_forms:
                        line_parts[1] = fixed_forms[(sent_id, node_id)]
                        fixed_line = '\t'.join(line_parts)
                        fixed_sent_lines.append(fixed_line)
                    else:
                        fixed_sent_lines.append(line)
                else:
                    fixed_sent_lines.append(line)
        sent_lines = '\n'.join(fixed_sent_lines)
        f.write(sent_lines)
        f.write('\n')

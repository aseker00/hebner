with open('data/clean/treebank/spmrl-05.conllu') as f:
    lines = f.readlines()
sentences = {}
sent_lines = []
for line in [l.strip() for l in lines]:
    sent_lines.append(line.replace('”', '"'))
    if not line:
        sentences[sent_id] = sent_lines
        sent_lines = []
    elif '# sent_id = ' in line:
        sent_id = int(line[len('# sent_id = '):])
with open('data/clean/treebank/spmrl-06.conllu', 'w') as f:
    for sent_id in sentences:
        sent_lines = '\n'.join(sentences[sent_id])
        f.write(sent_lines)
        f.write('\n')

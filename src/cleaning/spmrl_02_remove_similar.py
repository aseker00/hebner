import operator
from collections import defaultdict, Counter


sentences = {}
similar_ids = defaultdict(list)
similar_sets = Counter()
unique_ids = set()
with open('data/clean/treebank/spmrl-01.conllu') as f:
    lines = f.readlines()

sent_lines = []
for line in [l.strip() for l in lines]:
    sent_lines.append(line)
    if not line:
        sentences[sent_id] = sent_lines
        sent_lines = []
    elif '# sent_id = ' in line:
        sent_id = int(line[len('# sent_id = '):])
    elif '# very_similar_sent_id = ' in line:
        for part in [p for p in line[len('# very_similar_sent_id = '):][1:-1].split(', ') if p]:
            dup_id = int(part)
            similar_ids[sent_id].append(dup_id)
        if similar_ids[sent_id]:
            similar_sets[tuple(sorted([sent_id] + similar_ids[sent_id]))] += 1
ids_to_remove = set()
for sim_set in similar_sets:
    sim_lens = {}
    for sim_id in sim_set:
        sim_lens[sim_id] = len(sentences[sim_id][2])
    max_len_id = max(sim_lens.items(), key=operator.itemgetter(1))[0]
    ids_to_remove.update([sid for sid in sim_set if sid != max_len_id])
with open('data/clean/treebank/spmrl-02.conllu', 'w') as f:
    for sent_id in [sid for sid in sentences if sid not in ids_to_remove]:
        sent_lines = '\n'.join(sentences[sent_id])
        f.write(sent_lines)
        f.write('\n')

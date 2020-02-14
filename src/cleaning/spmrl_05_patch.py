import patch
from shutil import copyfile
src = 'data/clean/treebank/spmrl-04.conllu'
dst = 'data/clean/treebank/spmrl-05.conllu'
copyfile(src, dst)

pset = patch.fromfile('src/cleaning/spmrl_05_diff.patch')
pset.apply()

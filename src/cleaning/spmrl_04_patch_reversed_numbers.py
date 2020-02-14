import patch
from shutil import copyfile
src = 'data/clean/treebank/spmrl-03.conllu'
dst = 'data/clean/treebank/spmrl-04.conllu'
copyfile(src, dst)

pset = patch.fromfile('src/cleaning/spmrl_04_diff.patch')
pset.apply()

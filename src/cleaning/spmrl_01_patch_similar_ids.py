import patch
from shutil import copyfile
src = 'data/raw/treebank/spmrl_fixed.conllu'
dst = 'data/clean/treebank/spmrl-01.conllu'
copyfile(src, dst)

pset = patch.fromfile('src/cleaning/spmrl_01_diff.patch')
pset.apply()

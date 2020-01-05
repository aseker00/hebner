import numpy as np
from torch.nn.modules.sparse import EmbeddingBag
import torch
from fasttext import load_model
import gzip
from pathlib import Path


def _load_line(line: str, w2i: dict, i2v: torch.Tensor, idx: int, device) -> int:
    line = line.split()
    word = line[0]
    i2v[idx] = torch.tensor([float(v) for v in line[1:]], dtype=torch.float, device=device)
    w2i[word] = idx
    return idx + 1


def _file_lines(path: Path) -> int:
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            lines = f.readlines()
    else:
        with open(str(path), 'r') as f:
            lines = f.readlines()
    return len(lines)


def _vec_len(path: Path) -> int:
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            line = f.readline()
    else:
        with open(str(path), 'r') as f:
            line = f.readline()
    line = line.split()
    return len([float(v) for v in line[1:]])


def _load_file(path: Path, device, idx_offset) -> (dict, torch.Tensor):
    n_entries = _file_lines(path)
    n_dimensions = _vec_len(path)
    w2i = {}
    i2v = torch.zeros((n_entries + idx_offset, n_dimensions), dtype=torch.float, device=device)
    idx = idx_offset
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            for line in f:
                idx = _load_line(line, w2i, i2v, idx, device=device)
    else:
        with open(str(path), 'r') as f:
            for line in f:
                idx = _load_line(line, w2i, i2v, idx, device=device)
    return w2i, i2v


def embedding_weight_matrix(resource: str, device, idx_offset=0) -> (dict, torch.Tensor):
    print('\nLoading FT embedding layer: {}'.format(resource))
    return _load_file(Path('model/ft_{}.vec'.format(resource)), device, idx_offset)


class FastTextEmbeddingBag(EmbeddingBag):

    def __init__(self, path: str):
        print('\nLoading FT model')
        self.model = load_model(path)
        self.dimension = self.model.get_dimension()
        input_matrix = self.model.get_input_matrix()
        input_matrix_shape = input_matrix.shape
        super().__init__(input_matrix_shape[0], input_matrix_shape[1])
        self.weight.data.copy_(torch.tensor(input_matrix, dtype=torch.float))
        self.weight.requires_grad=False

    def forward(self, words):
        word_subinds = np.empty([0], dtype=np.int64)
        word_offsets = [0]
        for word in words:
            _, subinds = self.model.get_subwords(word)
            word_subinds = np.concatenate((word_subinds, subinds))
            word_offsets.append(word_offsets[-1] + len(subinds))
        word_offsets = word_offsets[:-1]
        ind = torch.tensor(word_subinds, dtype=torch.long)
        offsets = torch.tensor(word_offsets, dtype=torch.long)
        return super().forward(ind, offsets)

    def text_to_vector(self, text, device) -> torch.Tensor:
        words = text.split()
        x = np.zeros((len(words), self.dimension))
        for i, word in enumerate(words):
            x[i, :] = self.model.get_word_vector(word).astype('float32')
        return torch.tensor(x, dtype=torch.float, device=device)

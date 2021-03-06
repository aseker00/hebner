import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer


# labels = ['O', 'B-PER', 'B-LOC', 'B-ORG', 'B-TTL', 'I-PER', 'I-LOC', 'I-ORG', 'I-TTL']
labels = ['O', 'B-PER', 'B-LOC', 'B-ORG', 'I-PER', 'I-LOC', 'I-ORG']


class XfmrNerModel(nn.Module):

    def __init__(self, name: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                 dropout_prob: float = 0.0, classifier_input_feat_num: int = None):
        super(XfmrNerModel, self).__init__()
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(dropout_prob)
        self.labels = labels
        self.label2id = {label: label_id for label_id, label in enumerate(labels)}
        self.num_labels = len(self.label2id)
        self.classifier = nn.Linear(model.config.hidden_size if classifier_input_feat_num is None else
                                    classifier_input_feat_num, self.num_labels)
        self.max_seq_len = model.config.max_position_embeddings
        self._init_weights()

    @property
    def cls_token(self):
        return self.tokenizer.cls_token

    @property
    def sep_token(self):
        return self.tokenizer.sep_token

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.01)

    def tokenize(self, word: str) -> (list, list):
        tokens = self.tokenizer.tokenize(word)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids


    def forward(self, valid_index: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, label_ids: torch.Tensor = None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        valid_attention_mask = [attention_mask[i, valid_index[i] == 1][1:-1] for i in range(attention_mask.size(0))]
        valid_attention_mask = pad_sequence(valid_attention_mask, batch_first=True, padding_value=0)
        valid_labels = [label_ids[i, valid_index[i] == 1][1:-1] for i in range(label_ids.size(0))]
        valid_labels = pad_sequence(valid_labels, batch_first=True, padding_value=0)
        valid_logits = [logits[i, valid_index[i] == 1][1:-1] for i in range(logits.size(0))]
        valid_logits = pad_sequence(valid_logits, batch_first=True, padding_value=0)
        valid_outputs = valid_logits.argmax(dim=2)
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = valid_attention_mask.view(-1) == 1
                active_logits = valid_logits.view(-1, self.num_labels)[active_loss]
                active_labels = valid_labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(valid_logits.view(-1, self.num_labels), valid_labels.view(-1))
            return loss, valid_attention_mask, valid_labels, valid_outputs
        return valid_attention_mask, valid_labels, valid_outputs

    def to_sample(self, sent_idx: int, text: str, df: pd.DataFrame, max_token_seq_len: int) -> dict:
        sample = {'sent_idx': sent_idx, 'text': text}
        tokens_gb = df.groupby('token_idx')
        grouped_tokens = [tokens_gb.get_group(x) for x in tokens_gb.groups]
        tokens = [x['token'].to_numpy()[0] for x in grouped_tokens]
        token_start_offsets = [x['token_start_offset'].to_numpy()[0] for x in grouped_tokens]
        token_end_offsets = [x['token_end_offset'].to_numpy()[0] for x in grouped_tokens]
        sample['tokens'] = tokens
        sample['token_start_offsets'] = token_start_offsets
        sample['token_end_offsets'] = token_end_offsets
        subword_lengths = [len(x) for x in grouped_tokens]
        token_start_idxs = np.cumsum([0] + subword_lengths[:-1])
        # token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_end_idxs = np.cumsum(subword_lengths) - 1
        sample['xfmr_start_idx'] = np.zeros(max_token_seq_len, dtype=np.int)
        sample['xfmr_start_idx'][token_start_idxs] = 1
        sample['xfmr_end_idx'] = np.zeros(max_token_seq_len, dtype=np.int)
        sample['xfmr_end_idx'][token_end_idxs] = 1
        sample['xfmr_tokens'] = np.array([self.tokenizer.pad_token_id] * max_token_seq_len, dtype=np.int)
        sample['xfmr_tokens'][:len(df.index)] = df['xfmr_token_id'].to_numpy()
        # sample['xfmr_token_labels'] = np.array([self.label2id[self.tokenizer.pad_token]] * max_seq_len, dtype=np.int)
        sample['xfmr_token_labels'] = np.array([self.label2id[self.labels[0]]] * max_token_seq_len, dtype=np.int)
        sample['xfmr_token_labels'][:len(df.index)] = df['token_label_id'].to_numpy()
        sample['xfmr_attention_mask'] = np.zeros(max_token_seq_len, dtype=np.int)
        sample['xfmr_attention_mask'][:len(df.index)] = 1
        return sample

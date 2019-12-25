import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import pandas as pd
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer


labels = {'O', 'B-PER', 'B-LOC', 'B-ORG', 'I-PER', 'I-LOC', 'I-ORG'}


class XfmrNerModel(nn.Module):

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, hidden_dropout_prob=0.5):
        super(XfmrNerModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(hidden_dropout_prob)
        tokenizer_labels = {tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token}
        self.labels = list(reversed(sorted(labels.union(tokenizer_labels))))
        self.label2id = {label: label_id for label_id, label in enumerate(self.labels)}
        self.num_labels = len(self.label2id)
        self.classifier = nn.Linear(model.config.hidden_size, self.num_labels)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.01)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # loss_fct = NLLLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return None, logits

    def predict(self, text, tokenizer):
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        outputs = self.model(input_ids)

    def to_sample(self, sent_idx: int, df: pd.DataFrame, max_seq_len: int) -> dict:
        sample = {'sent_idx': sent_idx}
        tokens_gb = df.groupby('token_idx')
        grouped_tokens = [tokens_gb.get_group(x) for x in tokens_gb.groups]
        tokens = [x['token'].to_numpy()[0] for x in grouped_tokens]
        sample['tokens'] = tokens
        subword_lengths = [len(x) for x in grouped_tokens]
        token_start_idxs = np.cumsum([0] + subword_lengths[:-1])
        # token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_end_idxs = np.cumsum(subword_lengths) - 1
        sample['token_start_idx'] = np.zeros(max_seq_len, dtype=np.int)
        sample['token_start_idx'][token_start_idxs] = 1
        sample['token_end_idx'] = np.zeros(max_seq_len, dtype=np.int)
        sample['token_end_idx'][token_end_idxs] = 1
        sample['xfmr_tokens'] = np.array([self.tokenizer.pad_token_id] * max_seq_len, dtype=np.int)
        sample['xfmr_tokens'][:len(df.index)] = df['xfmr_token_id'].to_numpy()
        sample['xfmr_labels'] = np.array([self.label2id[self.tokenizer.pad_token]] * max_seq_len, dtype=np.int)
        sample['xfmr_labels'][:len(df.index)] = df['label_id'].to_numpy()
        sample['xfmr_attention_mask'] = np.zeros(max_seq_len, dtype=np.int)
        sample['xfmr_attention_mask'][:len(df.index)] = 1
        return sample

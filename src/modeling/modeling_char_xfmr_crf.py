from fasttext.FastText import _FastText
from torch.nn.utils.rnn import pad_sequence

from src.modeling.modeling_char_xfmr import CharXfmrNerModel
from src.modeling.modeling_token_xfmr import XfmrNerModel
from torchcrf import CRF
import torch


transition_matrix = {
    'SOS': ['B-PER', 'B-LOC', 'B-ORG', 'O'],
    'O': ['B-PER', 'B-LOC', 'B-ORG', 'O'],
    'B-PER': ['I-PER'],
    'B-LOC': ['I-LOC'],
    'B-ORG': ['I-ORG'],
    'I-PER': ['B-PER', 'B-LOC', 'B-ORG', 'I-PER', 'O'],
    'I-LOC': ['B-PER', 'B-LOC', 'B-ORG', 'I-LOC', 'O'],
    'I-ORG': ['B-PER', 'B-LOC', 'B-ORG', 'I-ORG', 'O'],
}


class CharXfmrCrfNerModel(CharXfmrNerModel):

    def __init__(self, x_model: XfmrNerModel, ft_model: _FastText, char2id: dict, char_dropout_prob: float = 0.0):
        super(CharXfmrCrfNerModel, self).__init__(x_model, ft_model, char2id, char_dropout_prob)
        self.crf = CRF(self.num_labels, batch_first=True)
        self._init_transition_weights()

    def _init_transition_weights(self):
        with torch.no_grad():
            sos_nil_transitions = [self.label2id[label] for label in self.labels if label not in transition_matrix['SOS']]
            for label_id in sos_nil_transitions:
                self.crf.start_transitions[label_id] = -1000
            for from_label in transition_matrix:
                if from_label == 'SOS':
                    continue
                from_label_id = self.label2id[from_label]
                from_nil_transitions = [self.label2id[label] for label in self.labels if label not in transition_matrix[from_label]]
                for label_id in from_nil_transitions:
                    self.crf.transitions[from_label_id, label_id] = -1000

    def forward(self, token_input_ids: torch.Tensor, token_attention_mask: torch.Tensor, char_token_idx: torch.Tensor,
                char_input_ids: torch.Tensor, char_attention_mask: torch.Tensor, char_label_ids: torch.Tensor = None):
        outputs = self.x_model.model(token_input_ids[:, :self.max_seq_len], attention_mask=token_attention_mask[:, :self.max_seq_len])
        sequence_output = outputs[0]
        sequence_output = self.x_model.dropout(sequence_output)
        embedded_chars = self.char_emb(char_input_ids)
        embedded_chars = self.char_dropout(embedded_chars)
        char_sequences = []
        for sent_idx in range(sequence_output.size(0)):
            sent_sequence_output = sequence_output[sent_idx]
            sent_char_token_idx = char_token_idx[sent_idx]
            sent_embedded_chars = embedded_chars[sent_idx]
            sent_token_outputs = [sent_sequence_output[token_idx] for token_idx in sent_char_token_idx]
            sent_sequence = torch.cat([torch.stack(sent_token_outputs, dim=0), sent_embedded_chars], dim=1)
            char_sequences.append(sent_sequence)
        char_sequences = torch.stack(char_sequences, dim=0)
        logits = self.classifier(char_sequences)
        valid_logits = [logits[i, char_attention_mask[i] == 1][1:-1] for i in range(logits.size(0))]
        valid_logits = pad_sequence(valid_logits, batch_first=True, padding_value=0)
        valid_labels = [char_label_ids[i, char_attention_mask[i] == 1][1:-1] for i in range(char_label_ids.size(0))]
        valid_labels = pad_sequence(valid_labels, batch_first=True, padding_value=0)
        valid_attention_mask = [char_attention_mask[i, char_attention_mask[i] == 1][1:-1] for i in range(char_attention_mask.size(0))]
        valid_attention_mask = pad_sequence(valid_attention_mask, batch_first=True, padding_value=0)
        valid_decoded_labels = torch.tensor(self.crf.decode(emissions=valid_logits), dtype=torch.long)
        if char_label_ids is not None:
            log_likelihood = self.crf(emissions=valid_logits, tags=valid_labels, mask=valid_attention_mask, reduction='mean')
            loss = -log_likelihood
            return loss, valid_attention_mask, valid_labels, valid_decoded_labels
        return valid_attention_mask, valid_labels, valid_decoded_labels

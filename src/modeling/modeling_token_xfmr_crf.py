import torch
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from transformers import PreTrainedTokenizer, PreTrainedModel
from src.modeling.modeling_token_xfmr import XfmrNerModel


transition_matrix = {
    'SOS': ['B-PER', 'B-LOC', 'B-ORG', 'O'],
    'O': ['B-PER', 'B-LOC', 'B-ORG', 'O'],
    'B-PER': ['B-PER', 'B-LOC', 'B-ORG', 'I-PER', 'O'],
    'B-LOC': ['B-PER', 'B-LOC', 'B-ORG', 'I-LOC', 'O'],
    'B-ORG': ['B-PER', 'B-LOC', 'B-ORG', 'I-ORG', 'O'],
    'I-PER': ['B-PER', 'B-LOC', 'B-ORG', 'I-PER', 'O'],
    'I-LOC': ['B-PER', 'B-LOC', 'B-ORG', 'I-LOC', 'O'],
    'I-ORG': ['B-PER', 'B-LOC', 'B-ORG', 'I-ORG', 'O'],
}


class XfmrCrfNerModel(XfmrNerModel):

    def __init__(self, name: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, hidden_dropout_prob: float = 0.0, classifier_input_feat_num: int = None):
        super(XfmrCrfNerModel, self).__init__(name, tokenizer, model, hidden_dropout_prob, classifier_input_feat_num)
        self.crf = CRF(self.num_labels, batch_first=True)
        self._init_transition_weights()

    def _init_transition_weights(self):
        with torch.no_grad():
            sos_nil_transitions = [self.label2id[label] for label in self.labels if label not in transition_matrix['SOS']]
            for label_id in sos_nil_transitions:
                self.crf.start_transitions[label_id] = -100000
            for from_label in transition_matrix:
                if from_label == 'SOS':
                    continue
                from_label_id = self.label2id[from_label]
                from_nil_transitions = [self.label2id[label] for label in self.labels if label not in transition_matrix[from_label]]
                for label_id in from_nil_transitions:
                    self.crf.transitions[from_label_id, label_id] = -100000

    def forward(self, input_mask: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.ByteTensor, label_ids: torch.Tensor = None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # masked_input = [input_ids[i, input_mask[i] == 1][1:-1] for i in range(input_ids.size(0))]
        # masked_input = pad_sequence(masked_input, batch_first=True, padding_value=0)
        masked_attention_mask = [attention_mask[i, input_mask[i] == 1][1:-1] for i in range(attention_mask.size(0))]
        masked_attention_mask = pad_sequence(masked_attention_mask, batch_first=True, padding_value=0)
        masked_labels = [label_ids[i, input_mask[i] == 1][1:-1] for i in range(label_ids.size(0))]
        masked_labels = pad_sequence(masked_labels, batch_first=True, padding_value=0)
        masked_logits = [logits[i, input_mask[i] == 1][1:-1] for i in range(logits.size(0))]
        masked_logits = pad_sequence(masked_logits, batch_first=True, padding_value=0)
        decoded_labels = torch.tensor(self.crf.decode(emissions=masked_logits), dtype=torch.long)
        if label_ids is not None:
            log_likelihood = self.crf(emissions=masked_logits, tags=masked_labels, mask=masked_attention_mask, reduction='mean')
            loss = -log_likelihood
            return loss, masked_attention_mask, masked_labels, decoded_labels
        return masked_labels, decoded_labels

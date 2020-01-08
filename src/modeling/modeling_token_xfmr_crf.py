import torch
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.ByteTensor, label_ids: torch.Tensor = None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        decoded_labels = torch.tensor(self.crf.decode(emissions=logits), dtype=torch.long)
        if label_ids is not None:
            log_likelihood = self.crf(emissions=logits, tags=label_ids, mask=attention_mask, reduction='mean')
            loss = -log_likelihood
            return loss, decoded_labels
        return decoded_labels

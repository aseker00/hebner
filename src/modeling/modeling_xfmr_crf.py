from src.modeling.modeling_xfmr import *
from torchcrf import CRF


class XfmrCrfNerModel(XfmrNerModel):

    def __init__(self, name: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, hidden_dropout_prob=0.5):
        super(XfmrCrfNerModel, self).__init__(name, tokenizer, model, hidden_dropout_prob)
        self.crf = CRF(self.num_labels, batch_first=True)

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
from fasttext.FastText import _FastText
from src.modeling.modeling_char_xfmr import CharXfmrNerModel
from src.modeling.modeling_xfmr import XfmrNerModel
from torchcrf import CRF
import torch


class CharXfmrCrfNerModel(CharXfmrNerModel):

    def __init__(self, x_model: XfmrNerModel, ft_model: _FastText, char2id: dict, char_dropout_prob: float = 0.5):
        super(CharXfmrCrfNerModel, self).__init__(x_model, ft_model, char2id, char_dropout_prob)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, token_input_ids: torch.Tensor, token_attention_mask: torch.Tensor, char_token_idx: torch.Tensor,
                char_input_ids: torch.Tensor, char_attention_mask: torch.Tensor, char_label_ids: torch.Tensor = None):
        x_token_outputs = self.x_model.model(token_input_ids[:, :self.max_seq_len],
                                             attention_mask=token_attention_mask[:, :self.max_seq_len])
        x_token_sequence_output = x_token_outputs[0]
        x_token_sequence_output = self.x_model.dropout(x_token_sequence_output)
        embedded_chars = self.char_emb(char_input_ids)
        embedded_chars = self.char_dropout(embedded_chars)
        sequence_outputs = []
        for sent_idx in range(x_token_sequence_output.size(0)):
            sent_token_output = x_token_sequence_output[sent_idx]
            sent_char_token_idx = char_token_idx[sent_idx]
            sent_embedded_chars = embedded_chars[sent_idx]
            sent_x_token_outputs = []
            for token_idx in sent_char_token_idx:
                x_token_output = sent_token_output[token_idx]
                sent_x_token_outputs.append(x_token_output)
            sent_sequence = torch.cat([torch.stack(sent_x_token_outputs, dim=0), sent_embedded_chars], dim=1)
            sequence_outputs.append(sent_sequence)
        sequence_outputs = torch.stack(sequence_outputs, dim=0)
        logits = self.classifier(sequence_outputs)
        decoded_labels = torch.tensor(self.crf.decode(emissions=logits), dtype=torch.long)
        if char_label_ids is not None:
            log_likelihood = self.crf(emissions=logits, tags=char_label_ids, mask=char_attention_mask, reduction='mean')
            loss = -log_likelihood
            return loss, decoded_labels
        return decoded_labels

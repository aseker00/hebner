from fasttext.FastText import _FastText
from src.modeling.modeling_xfmr import *


class CharXfmrNerModel(nn.Module):

    def __init__(self, x_model: XfmrNerModel, ft_model: _FastText, char2id: dict, char_dropout_prob=0.5):
        super(CharXfmrNerModel, self).__init__()
        self.x_model = x_model
        self.ft_model = ft_model
        self.char2id = char2id
        self.char_vectors = torch.tensor([ft_model.get_word_vector(c) for c in char2id], dtype=torch.float)
        self.char_emb = nn.Embedding.from_pretrained(self.char_vectors, freeze=False, padding_idx=0)
        self.char_dropout = nn.Dropout(char_dropout_prob)

    @property
    def name(self):
        return self.x_model.name

    @property
    def labels(self):
        return self.x_model.labels

    @property
    def label2id(self):
        return self.x_model.label2id

    @property
    def num_labels(self):
        return self.x_model.num_labels

    @property
    def cls_token(self):
        return self.x_model.cls_token

    @property
    def sep_token(self):
        return self.x_model.sep_token

    @property
    def pad_token(self):
        return self.x_model.pad_token

    @property
    def max_seq_len(self):
        return self.x_model.max_seq_len

    def forward(self, token_input_ids: torch.Tensor, token_attention_mask: torch.Tensor, char_token_idx: torch.Tensor,
                char_input_ids: torch.Tensor, char_attention_mask: torch.Tensor, char_label_ids: torch.Tensor = None):
        x_token_outputs = self.x_model.model(token_input_ids[:, :self.max_seq_len], attention_mask=token_attention_mask[:, :self.max_seq_len])
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
        logits = self.x_model.classifier(sequence_outputs)
        outputs = logits.argmax(dim=2).detach()
        if char_label_ids is not None:
            loss_fct = CrossEntropyLoss()
            if char_attention_mask is not None:
                active_loss = char_attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = char_label_ids.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), char_label_ids.view(-1))
            return loss, outputs
        return outputs

    def to_sample(self, sent_idx: int, text: str, token_df: pd.DataFrame, char_df: pd.DataFrame,
                  max_char_seq_len: int) -> dict:
        sent_tokens = token_df['xfmr_token']
        token_idx, char_idx = 0, 0
        cidx2xtidx = np.zeros(len(char_df.index), dtype=np.int)
        for token in sent_tokens:
            if token == self.cls_token or token == self.sep_token or token == self.pad_token:
                char_len = 1
            else:
                token_chars = list(token[:-4]) if token[-4:] == '</w>' else list(token)
                char_len = len(token_chars)
            cidx2xtidx[char_idx:char_idx+char_len] = token_idx
            char_idx += char_len
            token_idx += 1
        sample = self.x_model.to_sample(sent_idx, text, token_df, max_char_seq_len)
        sample['cidx2xtidx'] = np.zeros(max_char_seq_len, dtype=np.int)
        sample['cidx2xtidx'][:len(cidx2xtidx)] = cidx2xtidx
        sample['chars'] = np.array([self.x_model.tokenizer.pad_token_id] * max_char_seq_len, dtype=np.int)
        sample['chars'][:len(char_df.index)] = char_df['char_id'].to_numpy()
        sample['char_labels'] = np.array([self.label2id[self.labels[0]]] * max_char_seq_len, dtype=np.int)
        sample['char_labels'][:len(char_df.index)] = char_df['char_label_id'].to_numpy()
        sample['char_attention_mask'] = np.zeros(max_char_seq_len, dtype=np.int)
        sample['char_attention_mask'][:len(char_df.index)] = 1
        return sample

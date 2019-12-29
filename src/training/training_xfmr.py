import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.modeling.modeling_xfmr import XfmrNerModel
from src.training.training_utils import LabeledSentence, print_sample, print_metrics


# model = HebNER("xl")
# output = model.predict("Steve went to Paris")
# print(output)
# '''
#     [
#         {
#             "confidence": 0.9981840252876282,
#             "tag": "B-PER",
#             "word": "Steve"
#         },
#         {
#             "confidence": 0.9998939037322998,
#             "tag": "O",
#             "word": "went"
#         },
#         {
#             "confidence": 0.999891996383667,
#             "tag": "O",
#             "word": "to"
#         },
#         {
#             "confidence": 0.9991968274116516,
#             "tag": "B-LOC",
#             "word": "Paris"
#         }
#     ]
# '''


def to_dataset(samples: list) -> (TensorDataset, dict):
    sent_ids = torch.tensor([sample['sent_idx'] for sample in samples], dtype=torch.long)
    token_idx = torch.tensor([sample['xfmr_start_idx'] for sample in samples], dtype=torch.long)
    input_ids = torch.tensor([sample['xfmr_tokens'] for sample in samples], dtype=torch.long)
    label_ids = torch.tensor([sample['xfmr_labels'] for sample in samples], dtype=torch.long)
    attention_mask = torch.tensor([sample['xfmr_attention_mask'] for sample in samples], dtype=torch.bool)
    dataset = TensorDataset(sent_ids, token_idx, input_ids, label_ids, attention_mask)
    id2sample = {sample['sent_idx']: sample for sample in samples}
    return dataset, id2sample


class ModelOptimizer:

    def __init__(self, optimizer, scheduler, parameters, max_grad_norm):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = parameters
        self.max_grad_norm = max_grad_norm

    def step(self):
        nn.utils.clip_grad_norm_(parameters=self.parameters, max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()


def run_step(device, batch: tuple, model: XfmrNerModel, model_optimizer: ModelOptimizer) -> (torch.Tensor, torch.Tensor):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_label_ids, b_attention_mask = batch
    loss, pred_labels = model(b_input_ids, b_attention_mask, b_label_ids)
    if loss:
        loss.backward()
    if model_optimizer:
        model_optimizer.step()
    return loss, pred_labels


def decode_batch(batch: list, samples: dict, model: XfmrNerModel) -> (list, list):
    batch_sent_ids, batch_token_idx, batch_input_ids, batch_label_ids, batch_attention_mask, batch_prediction_ids = batch
    gold_labeled_sentences = []
    pred_labeled_sentences = []
    for i in range(batch_input_ids.size(0)):
        sent_id = batch_sent_ids[i]
        data_sample = samples[sent_id.item()]
        attention_mask = batch_attention_mask[i]
        gold_ids = batch_label_ids[i][attention_mask == 1]
        pred_ids = batch_prediction_ids[i][attention_mask == 1]
        token_idx = batch_token_idx[i][attention_mask == 1]
        gold_ids = gold_ids[token_idx == 1][1:-1]
        pred_ids = pred_ids[token_idx == 1][1:-1]
        tokens = data_sample['tokens'][1:-1]
        text = data_sample['text']
        token_offsets = {start_offset: end_offset for start_offset, end_offset in zip(data_sample['token_start_offsets'][1:-1], data_sample['token_end_offsets'][1:-1])}
        gold_labels = [model.labels[label_id] for label_id in gold_ids]
        pred_labels = [model.labels[label_id] for label_id in pred_ids]
        gold_labeled_sent = LabeledSentence(sent_id, text, token_offsets, tokens, gold_labels)
        pred_labeled_sent = LabeledSentence(sent_id, text, token_offsets, tokens, pred_labels)
        gold_labeled_sentences.append(gold_labeled_sent)
        pred_labeled_sentences.append(pred_labeled_sent)
    return gold_labeled_sentences, pred_labeled_sentences


def run(epoch, phase, device, data: DataLoader, samples: dict, model: XfmrNerModel, print_every, model_optimizer=None) -> (list, list, float):
    print_loss, total_loss = 0, 0
    print_gold, print_pred, decoded_gold, decoded_pred = [], [], [], []
    for i, batch in enumerate(data):
        step = i + 1
        batch_loss, batch_predictions = run_step(device, batch[2:], model, model_optimizer)
        # batch_predictions = batch_outputs.argmax(dim=2)
        gold_labeled_sentences, pred_labeled_sentences = decode_batch(batch + [batch_predictions], samples, model)
        print_gold.extend(gold_labeled_sentences)
        print_pred.extend(pred_labeled_sentences)
        decoded_gold.extend(gold_labeled_sentences)
        decoded_pred.extend(pred_labeled_sentences)
        if batch_loss:
            batch_size = batch[0].size(0)
            print_loss += batch_loss.item() / batch_size
            total_loss += batch_loss.item()/batch_size
        if step % print_every == 0:
            if print_loss:
                print('epoch: {}, {}: {}({}) loss: {}'.format(epoch, phase, step, len(decoded_pred), print_loss/print_every))
            print_sample(epoch, phase, step, print_gold[-1], print_pred[-1])
            print_metrics(epoch, phase, step, print_gold, print_pred, model)
            print_loss = 0
            print_gold, print_pred = [], []
    if total_loss:
        print('epoch: {}, {}: {}({}) loss: {}'.format(epoch, phase, 'total', len(decoded_pred), total_loss/len(data)))
    print_sample(epoch, phase, 'total', decoded_gold[-1], decoded_pred[-1])
    print_metrics(epoch, phase, 'total', decoded_gold, decoded_pred, model)
    return decoded_gold, decoded_pred, total_loss

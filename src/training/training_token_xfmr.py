import torch
from torch.utils.data import DataLoader, TensorDataset
from src.modeling.modeling_token_xfmr import XfmrNerModel
from src.processing.processing_utils import TokenLabeledSentence
from src.training.training_utils import print_sample, print_metrics, ModelOptimizer, print_muc_eval


def to_token_dataset(samples: list) -> (TensorDataset, dict):
    sent_ids = torch.tensor([sample['sent_idx'] for sample in samples], dtype=torch.long)
    token_idx = torch.tensor([sample['xfmr_start_idx'] for sample in samples], dtype=torch.long)
    token_input_ids = torch.tensor([sample['xfmr_tokens'] for sample in samples], dtype=torch.long)
    token_label_ids = torch.tensor([sample['xfmr_token_labels'] for sample in samples], dtype=torch.long)
    token_attention_mask = torch.tensor([sample['xfmr_attention_mask'] for sample in samples], dtype=torch.bool)
    token_dataset = TensorDataset(sent_ids, token_idx, token_input_ids, token_attention_mask, token_label_ids)
    id2sample = {sample['sent_idx']: sample for sample in samples}
    return token_dataset, id2sample


def run_token_step(device, batch: tuple, x_model: XfmrNerModel, model_optimizer: ModelOptimizer) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    batch = tuple(t.to(device) for t in batch)
    loss, valid_token_mask, valid_gold_token_labels, valid_pred_token_labels = x_model(*batch)
    if model_optimizer is not None:
        loss.backward()
        model_optimizer.step()
    return loss, valid_token_mask.cpu(), valid_gold_token_labels.cpu(), valid_pred_token_labels.cpu()


def decode_token_batch(batch: list, samples: dict, x_model: XfmrNerModel) -> (list, list):
    # sent_ids, token_idx, token_input_ids, token_attention_mask, token_label_ids, token_mask, token_gold_ids, token_prediction_ids = batch
    sent_ids, _, _, _, _, valid_token_mask, valid_token_gold_ids, valid_token_prediction_ids = batch
    gold_token_labeled_sentences = []
    pred_token_labeled_sentences = []
    for i in range(valid_token_gold_ids.size(0)):
        sent_id = sent_ids[i]
        sent_data_sample = samples[sent_id.item()]
        sent_attention_mask = valid_token_mask[i]
        sent_gold_ids = valid_token_gold_ids[i][sent_attention_mask == 1]
        sent_pred_ids = valid_token_prediction_ids[i][sent_attention_mask == 1]
        # sent_token_idx = token_idx[i][sent_attention_mask == 1]
        # sent_gold_ids = sent_gold_ids[sent_token_idx == 1][1:-1]
        # sent_pred_ids = sent_pred_ids[sent_token_idx == 1][1:-1]
        # tokens = data_sample['tokens'][1:-1]
        sent_text = sent_data_sample['text']
        sent_token_offsets = {start_offset: end_offset for start_offset, end_offset in zip(sent_data_sample['token_start_offsets'][1:-1], sent_data_sample['token_end_offsets'][1:-1])}
        sent_gold_labels = [x_model.labels[label_id] for label_id in sent_gold_ids]
        sent_pred_labels = [x_model.labels[label_id] for label_id in sent_pred_ids]
        gold_labeled_sent = TokenLabeledSentence(sent_id, sent_text, sent_token_offsets, sent_gold_labels)
        pred_labeled_sent = TokenLabeledSentence(sent_id, sent_text, sent_token_offsets, sent_pred_labels)
        gold_token_labeled_sentences.append(gold_labeled_sent)
        pred_token_labeled_sentences.append(pred_labeled_sent)
    return gold_token_labeled_sentences, pred_token_labeled_sentences


def run_token(epoch, phase, device, data: DataLoader, samples: dict, x_model: XfmrNerModel, print_every, model_optimizer=None) -> (list, list, float):
    print_token_loss, total_token_loss = 0, 0
    print_token_gold, print_token_pred, decoded_token_gold, decoded_token_pred = [], [], [], []
    for i, batch in enumerate(data):
        step = i + 1
        batch_loss, batch_valid_mask, batch_valid_gold, batch_valid_pred = run_token_step(device, batch[1:], x_model, model_optimizer)
        gold_token_labeled_sentences, pred_token_labeled_sentences = decode_token_batch(batch + [batch_valid_mask, batch_valid_gold, batch_valid_pred], samples, x_model)
        print_token_gold.extend(gold_token_labeled_sentences)
        print_token_pred.extend(pred_token_labeled_sentences)
        decoded_token_gold.extend(gold_token_labeled_sentences)
        decoded_token_pred.extend(pred_token_labeled_sentences)
        batch_size = batch[0].size(0)
        print_token_loss += batch_loss.item() / batch_size
        total_token_loss += batch_loss.item() / batch_size
        if step % print_every == 0:
            print('epoch: {}, {}: {}({}) token loss: {}'.format(epoch, phase, step, len(decoded_token_pred), print_token_loss/print_every))
            print_sample(epoch, phase, step, print_token_gold[-1], print_token_pred[-1])
            print_metrics(epoch, phase, step, print_token_gold, print_token_pred, x_model.labels[1:])
            # print_muc_eval(epoch, phase, step, print_token_gold, print_token_pred, x_model.label2id)
            print_token_loss = 0
            print_token_gold, print_token_pred = [], []
    print('epoch: {}, {}: {}({}) token loss: {}'.format(epoch, phase, 'total', len(decoded_token_pred), total_token_loss/len(data)))
    print_sample(epoch, phase, 'total', decoded_token_gold[-1], decoded_token_pred[-1])
    print_metrics(epoch, phase, 'total', decoded_token_gold, decoded_token_pred, x_model.labels[1:])
    print_muc_eval(epoch, phase, 'total', decoded_token_gold, decoded_token_pred, x_model.label2id)
    return decoded_token_gold, decoded_token_pred, total_token_loss

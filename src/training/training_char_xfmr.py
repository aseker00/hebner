import torch
from torch.utils.data import DataLoader, TensorDataset
from src.modeling.modeling_char_xfmr import CharXfmrNerModel
from src.processing.processing_utils import CharLabeledSentence
from src.training.training_utils import print_sample, print_metrics, ModelOptimizer, print_muc_eval


def to_char_dataset(samples: list) -> (TensorDataset, dict):
    sent_ids = torch.tensor([sample['sent_idx'] for sample in samples], dtype=torch.long)
    token_idx = torch.tensor([sample['xfmr_start_idx'] for sample in samples], dtype=torch.long)
    token_input_ids = torch.tensor([sample['xfmr_tokens'] for sample in samples], dtype=torch.long)
    token_attention_mask = torch.tensor([sample['xfmr_attention_mask'] for sample in samples], dtype=torch.bool)
    char_token_idx = torch.tensor([sample['cidx2xtidx'] for sample in samples], dtype=torch.long)
    char_input_ids = torch.tensor([sample['chars'] for sample in samples], dtype=torch.long)
    char_label_ids = torch.tensor([sample['char_labels'] for sample in samples], dtype=torch.long)
    char_attention_mask = torch.tensor([sample['char_attention_mask'] for sample in samples], dtype=torch.bool)
    char_dataset = TensorDataset(sent_ids, token_idx, token_input_ids, token_attention_mask, char_token_idx, char_input_ids, char_attention_mask, char_label_ids)
    id2sample = {sample['sent_idx']: sample for sample in samples}
    return char_dataset, id2sample


def run_char_step(device, batch: tuple, cx_model: CharXfmrNerModel, model_optimizer: ModelOptimizer) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    batch = tuple(t.to(device) for t in batch)
    loss, valid_char_mask, valid_gold_char_labels, valid_pred_char_labels = cx_model(*batch)
    if model_optimizer is not None:
        loss.backward()
        model_optimizer.step()
    return loss, valid_char_mask.cpu(), valid_gold_char_labels.cpu(), valid_pred_char_labels.cpu()


def decode_char_batch(batch: list, samples: dict, cx_model: CharXfmrNerModel) -> (list, list):
    sent_ids, _, _, _, _, _, _, _, valid_char_mask, valid_char_gold_ids, valid_char_prediction_ids = batch
    gold_char_labeled_sentences = []
    pred_char_labeled_sentences = []
    for i in range(valid_char_gold_ids.size(0)):
        sent_id = sent_ids[i]
        sent_data_sample = samples[sent_id.item()]
        sent_attention_mask = valid_char_mask[i]
        sent_gold_ids = valid_char_gold_ids[i][sent_attention_mask == 1]
        sent_pred_ids = valid_char_prediction_ids[i][sent_attention_mask == 1]
        sent_text = sent_data_sample['text']
        sent_token_offsets = {start_offset: end_offset for start_offset, end_offset in zip(sent_data_sample['token_start_offsets'][1:-1], sent_data_sample['token_end_offsets'][1:-1])}
        sent_gold_labels = [cx_model.labels[label_id] for label_id in sent_gold_ids]
        sent_pred_labels = [cx_model.labels[label_id] for label_id in sent_pred_ids]
        gold_labeled_sent = CharLabeledSentence(sent_id, sent_text, sent_token_offsets, sent_gold_labels)
        pred_labeled_sent = CharLabeledSentence(sent_id, sent_text, sent_token_offsets, sent_pred_labels)
        gold_char_labeled_sentences.append(gold_labeled_sent)
        pred_char_labeled_sentences.append(pred_labeled_sent)
    return gold_char_labeled_sentences, pred_char_labeled_sentences


def run_char(epoch, phase, device, data: DataLoader, samples: dict, cx_model: CharXfmrNerModel, print_every, model_optimizer=None) -> (list, list, float):
    print_char_loss, total_char_loss = 0, 0
    print_char_gold, print_char_pred, decoded_char_gold, decoded_char_pred = [], [], [], []
    for i, batch in enumerate(data):
        step = i + 1
        batch_loss, batch_valid_mask, batch_valid_gold, batch_valid_pred = run_char_step(device, batch[2:], cx_model, model_optimizer)
        gold_char_labeled_sentences, pred_char_labeled_sentences = decode_char_batch(batch + [batch_valid_mask, batch_valid_gold, batch_valid_pred], samples, cx_model)
        print_char_gold.extend(gold_char_labeled_sentences)
        print_char_pred.extend(pred_char_labeled_sentences)
        decoded_char_gold.extend(gold_char_labeled_sentences)
        decoded_char_pred.extend(pred_char_labeled_sentences)
        batch_size = batch[0].size(0)
        print_char_loss += batch_loss.item() / batch_size
        total_char_loss += batch_loss.item()/batch_size
        if step % print_every == 0:
            print('epoch: {}, {}: {}({}) char loss: {}'.format(epoch, phase, step, len(decoded_char_pred), print_char_loss/print_every))
            print_sample(epoch, phase, step, print_char_gold[-1], print_char_pred[-1])
            print_metrics(epoch, phase, step, print_char_gold, print_char_pred, cx_model.labels[1:])
            print_char_loss = 0
            print_char_gold, print_char_pred = [], []
    print('epoch: {}, {}: {}({}) char loss: {}'.format(epoch, phase, 'total', len(decoded_char_pred), total_char_loss/len(data)))
    print_sample(epoch, phase, 'total', decoded_char_gold[-1], decoded_char_pred[-1])
    print_metrics(epoch, phase, 'total', decoded_char_gold, decoded_char_pred, cx_model.labels[1:])
    print_muc_eval(epoch, phase, 'total', decoded_char_gold, decoded_char_pred, cx_model.label2id)
    return decoded_char_gold, decoded_char_pred, total_char_loss

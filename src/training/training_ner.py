import torch
import torch.nn as nn
import pandas as pd
from seqeval.metrics import *
from sklearn.metrics import classification_report
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLMTokenizer, XLMModel, AdamW, get_linear_schedule_with_warmup
from src.modeling.modeling_xfmr import XfmrNerModel
from src.processing.processing_utils import load_model_data_samples


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

xlm_tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
xlm_model = XLMModel.from_pretrained('xlm-mlm-100-1280')
ner_model = XfmrNerModel(xlm_tokenizer, xlm_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    ner_model.cuda(device)


def to_dataset(samples: list) -> (TensorDataset, dict):
    sent_ids = torch.tensor([sample['sent_idx'] for sample in samples], dtype=torch.long)
    token_idx = torch.tensor([sample['token_start_idx'] for sample in samples], dtype=torch.long)
    input_ids = torch.tensor([sample['xfmr_tokens'] for sample in samples], dtype=torch.long)
    label_ids = torch.tensor([sample['xfmr_labels'] for sample in samples], dtype=torch.long)
    attention_mask = torch.tensor([sample['xfmr_attention_mask'] for sample in samples], dtype=torch.bool)
    dataset = TensorDataset(sent_ids, token_idx, input_ids, label_ids, attention_mask)
    id2sample = {sample['sent_idx']: sample for sample in samples}
    return dataset, id2sample


train_samples = load_model_data_samples('train')
train_dataset, train_samples = to_dataset(train_samples)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)

valid_samples = load_model_data_samples('valid')
valid_dataset, valid_samples = to_dataset(valid_samples)
valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=8)

test_samples = load_model_data_samples('test')
test_dataset, test_samples = to_dataset(test_samples)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=8)


def train_step(batch: tuple, model: XfmrNerModel, model_optimizer: Optimizer, model_optimizer_scheduler, model_parameters, max_norm) -> (torch.Tensor, torch.Tensor):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_label_ids, b_attention_mask = batch
    loss, outputs = model(b_input_ids, b_attention_mask, b_label_ids)
    loss.backward()
    nn.utils.clip_grad_norm_(parameters=model_parameters, max_norm=max_norm)
    model_optimizer.step()
    model_optimizer_scheduler.step()
    model_optimizer.zero_grad()
    return loss, outputs.cpu()


def print_sample(gold_sent, pred_sent):
    print('Tokens: {}'.format(' '.join(gold_sent[0])))
    print('Gold labels: {}'.format(' '.join(gold_sent[1])))
    print('Predicted labels: {}'.format(' '.join(pred_sent[1])))


def print_metrics(epoch, step, gold_sentences: list, pred_sentences: list, model: XfmrNerModel):
    pred_labels = [pred_id for sent in pred_sentences for pred_id in sent[1]]
    true_labels = [gold_id for sent in gold_sentences for gold_id in sent[1]]
    precision_micro = precision_score(true_labels, pred_labels)
    recall_micro = recall_score(true_labels, pred_labels)
    f1_micro = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    y_true = pd.Series(true_labels)
    y_pred = pd.Series(pred_labels)
    cross_tab = pd.crosstab(y_true, y_pred, rownames=['Real Label'], colnames=['Prediction'], margins=True)
    report = classification_report(y_true, y_pred, labels=model.labels[1:-2], target_names=model.labels[1:-2])
    print("epoch: {}, step: {}: precision={}, recall={}, f1={}, acc={}".format(epoch, step, precision_micro, recall_micro, f1_micro, accuracy))
    print(cross_tab)
    print(report)


def decode_batch(batch: list, samples: dict, model: XfmrNerModel) -> (list, list):
    batch_sent_ids, batch_token_idx, batch_input_ids, batch_label_ids, batch_attention_mask, batch_prediction_ids = batch
    gold_sentences = []
    pred_sentences = []
    for i in range(batch_input_ids.size(0)):
        sent_id = batch_sent_ids[i]
        data_sample = samples[sent_id.item()]
        attention_mask = batch_attention_mask[i]
        label_ids = batch_label_ids[i][attention_mask == 1]
        prediction_ids = batch_prediction_ids[i][attention_mask == 1]
        token_idx = batch_token_idx[i][attention_mask == 1]
        label_ids = label_ids[token_idx == 1][1:-1]
        prediction_ids = prediction_ids[token_idx == 1][1:-1]
        tokens = data_sample['tokens'][1:-1]
        gold_labels = [model.labels[label_id] for label_id in label_ids]
        pred_labels = [model.labels[label_id] for label_id in prediction_ids]
        gold_sentences.append((tokens, gold_labels, label_ids))
        pred_sentences.append((tokens, pred_labels, prediction_ids))
    return gold_sentences, pred_sentences


def train(epoch, data: DataLoader, samples: dict, model: XfmrNerModel, model_optimizer: Optimizer, model_optimizer_scheduler, model_parameters, max_norm, print_every):
    total_loss = 0
    total_gold = []
    total_pred = []
    print_loss = 0
    print_gold = []
    print_pred = []
    model.train()
    for step, batch in enumerate(data):
        batch_loss, batch_outputs = train_step(batch[2:], model, model_optimizer, model_optimizer_scheduler, model_parameters, max_norm)
        batch_predictions = batch_outputs.argmax(dim=2)
        gold_sentences, pred_sentences = decode_batch(batch + [batch_predictions], samples, model)
        total_gold.extend(gold_sentences)
        total_pred.extend(pred_sentences)
        print_gold.extend(gold_sentences)
        print_pred.extend(pred_sentences)
        batch_size = batch[0].size(0)
        total_loss += batch_loss.item()/batch_size
        print_loss += batch_loss.item()/batch_size
        if step > 0 and step % print_every == 0:
            print_sample(print_gold[-1], print_pred[-1])
            print_metrics(epoch, step, print_gold, print_pred, model)
            print_loss = 0
            print_gold = []
            print_pred = []
    print_sample(total_gold[-1], total_pred[-1])
    print_metrics(epoch, 'total', total_gold, total_pred, model)


epochs = 3
max_grad_norm = 1.0
lr = 1e-5
train_bsz = train_dataloader.batch_size
valid_bsz = valid_dataloader.batch_size
test_bsz = test_dataloader.batch_size
num_training_steps = len(train_dataloader.dataset) * epochs / train_dataloader.batch_size
num_warmup_steps = num_training_steps/10
optimizer = AdamW(ner_model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
# loss_function = nn.NLLLoss()
print_every = 1
for epoch in range(epochs):
    train(epoch, train_dataloader, train_samples, ner_model, optimizer, scheduler, ner_model.parameters(), max_grad_norm, print_every)

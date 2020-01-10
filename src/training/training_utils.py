import errno
import json
import os
import subprocess
import pandas as pd
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from torch import nn
from src.training.score_utils import get_chunks, score_sequence


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


class ModelOptimizer:

    def __init__(self, step_every, optimizer, scheduler, parameters, max_grad_norm):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = parameters
        self.max_grad_norm = max_grad_norm
        self.step_every = step_every
        self.steps = 0

    def step(self):
        self.steps += 1
        if self.steps % self.step_every == 0:
            nn.utils.clip_grad_norm_(parameters=self.parameters, max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()


def print_sample(epoch, phase, step, gold_sent, pred_sent):
    print('epoch: {}, {}: {} Sentence ID     : {}'.format(epoch, phase, step, gold_sent.sent_id))
    print('epoch: {}, {}: {} Tokens          : {}'.format(epoch, phase, step, ' '.join(gold_sent.tokens)))
    print('epoch: {}, {}: {} Gold labels     : {}'.format(epoch, phase, step, ' '.join(gold_sent.labels)))
    print('epoch: {}, {}: {} Predicted labels: {}'.format(epoch, phase, step, ' '.join(pred_sent.labels)))


def print_metrics(epoch, phase, step, gold_sentences: list, pred_sentences: list, labels: list):
    true_labels = [label for sent in gold_sentences for label in sent.labels]
    pred_labels = [label for sent in pred_sentences for label in sent.labels]
    precision_micro = precision_score(true_labels, pred_labels)
    recall_micro = recall_score(true_labels, pred_labels)
    f1_micro = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    y_true = pd.Series(true_labels)
    y_pred = pd.Series(pred_labels)
    cross_tab = pd.crosstab(y_true, y_pred, rownames=['Real Label'], colnames=['Prediction'], margins=True)
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels)
    print("epoch: {}, {}: {}, precision={}, recall={}, f1={}, acc={}".format(epoch, phase, step, precision_micro, recall_micro, f1_micro, accuracy))
    print(cross_tab)
    print(report)


def print_muc_eval(epoch, phase, step, gold_sentences: list, pred_sentences: list, label2id: dict):
    scores = None
    for gold_sentence, pred_sentence in zip(gold_sentences, pred_sentences):
        gold_label_ids = [label2id[label] for label in gold_sentence.labels]
        pred_label_ids = [label2id[label] for label in pred_sentence.labels]
        scores = score_sequence(get_chunks(gold_label_ids, label2id), get_chunks(pred_label_ids, label2id), scores=scores)
    print("epoch: {}, {}: {}, muc_eval={}".format(epoch, phase, step, json.dumps(scores.to_dict(), indent=1)))


def mkdir(folder_path: str):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def bash_command(cmd):
    subprocess.Popen(cmd, shell=True, executable='/bin/bash')


muc_eval_cmdline = 'java -Xmx4g -jar {}/corpuscmd-85.2.18.c61.0-stand-alone.jar CharLevelMucEval --tags "PER,LOC,ORG" --referenceData {} --testData {}; mv {}/agreement_output.txt {}/{}-{}_output.txt'

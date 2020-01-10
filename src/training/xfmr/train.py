import json
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from src.modeling.modeling_ner import create_char_ner_model, create_token_ner_model
from src.processing.processing_utils import load_model_data_samples, get_chars_from_processed_data
from src.training.training_char_xfmr import to_char_dataset, run_char
from src.training.training_token_xfmr import to_token_dataset, run_token
from src.training.training_utils import ModelOptimizer, mkdir, muc_eval_cmdline, bash_command

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    gpe_label = 'gpe-loc'
    xfmr_model_type = 'xlm'
    ner_model_type = 'char'
    classifier_type = 'crf'
    train_dataset_name = '{}-{}-{}-{}'.format('spmrl', ner_model_type, xfmr_model_type, gpe_label)
    train_sample_file_path = Path('data/processed/{}.pkl'.format(train_dataset_name))
    valid_dataset_name = '{}-{}-{}'.format('news', ner_model_type, xfmr_model_type)
    valid_sample_file_path = Path('data/processed/{}.pkl'.format(valid_dataset_name))
    test_dataset_name = '{}-{}-{}'.format('fin', ner_model_type, xfmr_model_type)
    test_sample_file_path = Path('data/processed/{}.pkl'.format(test_dataset_name))
    if ner_model_type == 'char':
        train_data_file_path = Path('data/processed/{}.csv'.format(train_dataset_name))
        valid_data_file_path = Path('data/processed/{}.csv'.format(valid_dataset_name))
        test_data_file_path = Path('data/processed/{}.csv'.format(test_dataset_name))
        data_files = [train_data_file_path, valid_data_file_path, test_data_file_path]
        data_chars = get_chars_from_processed_data(data_files)
        ft_model_path = Path("model/ft/cc.he.300.bin")
        ner_model = create_char_ner_model(data_chars, 0.3, xfmr_model_type, 0.3, classifier_type, ft_model_path)
    else:
        ner_model = create_token_ner_model(xfmr_model_type, 0.3, classifier_type)
    if torch.cuda.is_available():
        ner_model.cuda(device)
    print_every = 1
    epochs = 3
    lr = 1e-3
    parameters = list(ner_model.classifier.parameters()) + list(ner_model.crf.parameters())
    # parameters = ner_model.parameters()
    max_grad_norm = 1.0
    train_batch_size = 8
    eval_batch_size = 8
    for i in range(epochs):
        ner_model.train()
        epoch = i + 1
        train_samples = load_model_data_samples(train_sample_file_path)
        num_training_steps = len(train_samples) / train_batch_size
        num_warmup_steps = num_training_steps / 10
        optimizer = AdamW(parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        ner_model_optimizer = ModelOptimizer(32/train_batch_size, optimizer, scheduler, parameters, max_grad_norm)
        if ner_model_type == 'char':
            train_dataset, train_samples = to_char_dataset(train_samples)
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
            run_char(epoch, 'train', device, train_dataloader, train_samples, ner_model, print_every, ner_model_optimizer)
        else:
            train_dataset, train_samples = to_token_dataset(train_samples)
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
            run_token(epoch, 'train', device, train_dataloader, train_samples, ner_model, print_every, ner_model_optimizer)
        with torch.no_grad():
            ner_model.eval()
            for project_type in ['news', 'fin']:
                torch.cuda.empty_cache()
                phase = 'valid' if project_type == 'news' else 'test'
                samples = load_model_data_samples(valid_sample_file_path) if project_type == 'news' else load_model_data_samples(test_sample_file_path)
                if ner_model_type == 'char':
                    dataset, samples = to_char_dataset(samples)
                    sampler = SequentialSampler(dataset)
                    dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)
                    decoded_gold, decoded_pred, loss = run_char(epoch, phase, device, dataloader, samples, ner_model, print_every)
                else:
                    dataset, samples = to_token_dataset(samples)
                    sampler = SequentialSampler(dataset)
                    dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)
                    decoded_gold, decoded_pred, loss = run_token(epoch, phase, device, dataloader, samples, ner_model, print_every)
                gold_adms = [sent.to_adm() for sent in decoded_gold]
                gold_dir = '{}/{}/{}'.format('test', project_type, 'gold')
                mkdir(gold_dir)
                for sent, annotation in zip(decoded_gold, gold_adms):
                    with open('{}/{}.adm.json'.format(gold_dir, sent.sent_id), 'w') as outfile:
                        json.dump(annotation, outfile)
                pred_adms = [sent.to_adm() for sent in decoded_pred]
                pred_dir = '{}/{}/{}'.format('test', project_type, 'pred')
                mkdir(pred_dir)
                for sent, annotation in zip(decoded_pred, pred_adms):
                    with open('{}/{}.adm.json'.format(pred_dir, sent.sent_id), 'w') as outfile:
                        json.dump(annotation, outfile)
                cmd = muc_eval_cmdline.format('{}/{}'.format('.', 'test'), gold_dir, pred_dir,
                                              '.', '{}/{}'.format('.', 'test'),
                                              project_type, epoch)
                bash_command(cmd)


if __name__ == "__main__":
    main()

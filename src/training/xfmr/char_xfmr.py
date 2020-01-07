import json
from pathlib import Path
import fasttext
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import XLMTokenizer, XLMModel, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from src.modeling.modeling_char_xfmr import *
from src.processing.processing_utils import load_char_model_data_samples, load_processed_dataset
from src.training.training_char_xfmr import to_char_dataset, run_char
from src.training.training_utils import ModelOptimizer, mkdir, muc_eval_cmdline, bash_command

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(model_type: str = 'xlm'):
    gpe_label = 'LOC'
    train_dataset_name = '{}-{}-{}-{}'.format('spmrl', 'char', model_type, 'gpe-loc' if gpe_label == 'LOC' else 'gpe-org')
    train_char_data_file_path = Path('data/processed/{}.csv'.format(train_dataset_name))
    train_sample_file_path = Path('data/processed/{}.pkl'.format(train_dataset_name))
    valid_dataset_name = '{}-{}-{}'.format('news', 'char', model_type)
    valid_char_data_file_path = Path('data/processed/{}.csv'.format(valid_dataset_name))
    valid_sample_file_path = Path('data/processed/{}.pkl'.format(valid_dataset_name))
    test_dataset_name = '{}-{}-{}'.format('fin', 'char', model_type)
    test_char_data_file_path = Path('data/processed/{}.csv'.format(test_dataset_name))
    test_sample_file_path = Path('data/processed/{}.pkl'.format(test_dataset_name))

    train_samples = load_char_model_data_samples(train_sample_file_path)
    train_dataset, train_samples = to_char_dataset(train_samples)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)
    valid_samples = load_char_model_data_samples(valid_sample_file_path)
    valid_dataset, valid_samples = to_char_dataset(valid_samples)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=8)
    test_samples = load_char_model_data_samples(test_sample_file_path)
    test_dataset, test_samples = to_char_dataset(test_samples)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=8)


    train_df = load_processed_dataset(train_char_data_file_path)
    train_char2id = {a[0]: a[1] for a in train_df[['char', 'char_id']].to_numpy()}
    valid_df = load_processed_dataset(valid_char_data_file_path)
    valid_char2id = {a[0]: a[1] for a in valid_df[['char', 'char_id']].to_numpy()}
    test_df = load_processed_dataset(test_char_data_file_path)
    test_char2id = {a[0]: a[1] for a in test_df[['char', 'char_id']].to_numpy()}
    char2id = dict(dict(test_char2id, **valid_char2id), **train_char2id)

    ft_model = fasttext.load_model("model/ft/cc.he.300.bin")
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    x_model = XfmrNerModel(model_type, tokenizer, model, model.config.hidden_size + ft_model.get_dimension())
    char2id[x_model.pad_token] = 0
    char2id = {k: v for k, v in sorted(char2id.items(), key=lambda item: item[1])}
    ner_model = CharXfmrNerModel(x_model, ft_model, char2id)
    if torch.cuda.is_available():
        ner_model.cuda(device)
    lr = 1e-5
    optimizer = AdamW(ner_model.parameters(), lr=lr)
    num_training_steps = len(train_dataloader.dataset) / train_dataloader.batch_size
    num_warmup_steps = num_training_steps / 10
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    max_grad_norm = 1.0
    ner_model_optimizer = ModelOptimizer(optimizer, scheduler, ner_model.parameters(), max_grad_norm)
    print_every = 1
    epochs = 3
    for i in range(epochs):
        epoch = i + 1
        ner_model.train()
        run_char(epoch, 'train', device, train_dataloader, train_samples, ner_model, print_every, ner_model_optimizer)
        ner_model.eval()
        with torch.no_grad():
            for project_type in ['news', 'fin']:
                torch.cuda.empty_cache()
                phase = 'valid' if project_type == 'news' else 'test'
                dataloader = valid_dataloader if project_type == 'news' else test_dataloader
                samples = valid_samples if project_type == 'news' else test_samples
                decoded_gold, decoded_pred, loss = run_char(epoch, phase, device, dataloader, samples, ner_model, print_every)
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

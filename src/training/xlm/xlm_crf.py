import json
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import XLMTokenizer, XLMModel, AdamW, get_linear_schedule_with_warmup
from src.modeling.modeling_xfmr_crf import XfmrCrfNerModel
from src.processing.processing_utils import load_model_data_samples
from src.training.training_utils import mkdir, bash_command, muc_eval_cmdline
from src.training.training_xfmr import to_dataset, ModelOptimizer, run

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    xlm_tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
    xlm_model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    ner_model = XfmrCrfNerModel('xlm', xlm_tokenizer, xlm_model)
    if torch.cuda.is_available():
        ner_model.cuda(device)
    epochs = 3
    max_grad_norm = 1.0
    lr = 1e-5
    train_samples = load_model_data_samples('spmrl', ner_model)
    train_dataset, train_samples = to_dataset(train_samples)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)
    valid_samples = load_model_data_samples('news', ner_model)
    valid_dataset, valid_samples = to_dataset(valid_samples)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=8)
    test_samples = load_model_data_samples('fin', ner_model)
    test_dataset, test_samples = to_dataset(test_samples)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=8)
    num_training_steps = len(train_dataloader.dataset) * epochs / train_dataloader.batch_size
    num_warmup_steps = num_training_steps/10
    optimizer = AdamW(ner_model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    # loss_function = nn.NLLLoss()
    ner_model_optimizer = ModelOptimizer(optimizer, scheduler, ner_model.parameters(), max_grad_norm)
    print_every = 1
    for i in range(epochs):
        epoch = i + 1
        ner_model.train()
        run(epoch, 'train', device, train_dataloader, train_samples, ner_model, print_every, ner_model_optimizer)
        ner_model.eval()
        for project_type in ['news', 'fin']:
            phase = 'valid' if project_type == 'news' else 'test'
            dataloader = valid_dataloader if project_type == 'news' else test_dataloader
            samples = valid_samples if project_type == 'news' else test_samples
            decoded_gold, decoded_pred, loss = run(epoch, phase, device, dataloader, samples, ner_model, print_every)
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
            bash_command(muc_eval_cmdline.format(gold_dir, pred_dir, project_type, epoch))


if __name__ == "__main__":
    main()

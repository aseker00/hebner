from pathlib import Path
import fasttext
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import XLMTokenizer, XLMModel, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from src.modeling.modeling_char_xfmr import *
from src.processing.processing_utils import load_char_model_data_samples, load_processed_dataset
from src.training.training_char_xfmr import to_char_dataset, run_char
from src.training.training_utils import ModelOptimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(model_type: str = 'xlm'):
    ft_model = fasttext.load_model("model/ft/cc.he.300.bin")
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    x_model = XfmrNerModel(model_type, tokenizer, model, model.config.hidden_size + ft_model.get_dimension())
    train_samples = load_char_model_data_samples('.', 'spmrl', model_type)
    train_dataset, train_samples = to_char_dataset(train_samples)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)
    epochs = 3
    num_training_steps = len(train_dataloader.dataset) * epochs / train_dataloader.batch_size
    num_warmup_steps = num_training_steps/10
    lr = 1e-5
    char_data_file_path = Path('data/processed/{}-{}-{}.csv'.format('spmrl', model_type, 'char'))
    train_df = load_processed_dataset(char_data_file_path)
    char2id = {a[0]: a[1] for a in train_df[['char', 'char_id']].to_numpy()}
    char2id[x_model.pad_token] = 0
    ner_model = CharXfmrNerModel(x_model, ft_model, char2id)
    if torch.cuda.is_available():
        ner_model.cuda(device)
    optimizer = AdamW(ner_model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    max_grad_norm = 1.0
    ner_model_optimizer = ModelOptimizer(optimizer, scheduler, ner_model.parameters(), max_grad_norm)
    print_every = 1
    for i in range(epochs):
        epoch = i + 1
        ner_model.train()
        run_char(epoch, 'train', device, train_dataloader, train_samples, ner_model, print_every, ner_model_optimizer)


if __name__ == "__main__":
    main()

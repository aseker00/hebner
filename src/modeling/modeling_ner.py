from transformers import XLMTokenizer, XLMModel, BertTokenizer, BertModel
from src.modeling.modeling_token_xfmr import XfmrNerModel
from src.modeling.modeling_token_xfmr_crf import XfmrCrfNerModel
from src.modeling.modeling_char_xfmr import CharXfmrNerModel
from src.modeling.modeling_char_xfmr_crf import CharXfmrCrfNerModel
import fasttext
from pathlib import Path


def create_token_ner_model(xfmr_model_type: str, xfmr_dropout: float, classifier_type: str):
    if xfmr_model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    if classifier_type == 'crf':
        return  XfmrCrfNerModel(xfmr_model_type, tokenizer, model, xfmr_dropout)
    return XfmrNerModel(xfmr_model_type, tokenizer, model, xfmr_dropout)


def create_char_ner_model(chars: list, char_dropout: float, xfmr_model_type: str, xfmr_dropout: float, classifier_type: str, ft_model_file_path: Path):
    ft_model = fasttext.load_model(str(ft_model_file_path))
    if xfmr_model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    x_model = XfmrNerModel(xfmr_model_type, tokenizer, model, xfmr_dropout, model.config.hidden_size + ft_model.get_dimension())
    sorted_chars = chars.copy()
    for char in [x_model.sep_token, x_model.cls_token, x_model.pad_token]:
        if char not in sorted_chars:
            sorted_chars.insert(0, char)
    char2id = {c: i for i, c in enumerate(sorted_chars)}
    # char2id = {k: v for k, v in sorted(char2id.items(), key=lambda item: item[1])}
    if classifier_type == 'crf':
        return CharXfmrCrfNerModel(x_model, ft_model, char2id, char_dropout)
    return CharXfmrNerModel(x_model, ft_model, char2id, char_dropout)

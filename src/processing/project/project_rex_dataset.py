from pathlib import Path
from transformers import XLMTokenizer, XLMModel, BertTokenizer, BertModel
from src.modeling.modeling_token_xfmr import XfmrNerModel
from src.processing import processing_adm as adm
from src.processing.processing_utils import process_rex_labeled_sentences, save_processed_dataset
from src.processing.project.project_token_xfmr_dataset import label_token_sentences


def main(model_type: str = 'xlm'):
    project_sentences = {}
    for project_type in ['news', 'fin']:
        rex_sentences = adm.read_project_sentences(Path('data/clean/project-{}/{}'.format('rex', project_type)))
        rex_labeled_sentences = label_token_sentences(rex_sentences, normalize_rex)
        project_sentences[project_type] = rex_labeled_sentences

    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    ner_model = XfmrNerModel(model_type, tokenizer, model)

    for project_type in ['news', 'fin']:
        rex_labeled_sentences = project_sentences[project_type]
        df = process_rex_labeled_sentences(rex_labeled_sentences, ner_model)
        dataset_name = '{}-{}-{}'.format(project_type, model_type, 'rex')
        rex_data_file_path = Path('data/processed/{}.csv'.format(dataset_name))
        save_processed_dataset(df, rex_data_file_path)


if __name__ == "__main__":
    main()

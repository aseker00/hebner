from src.processing.project.project_xfmr_dataset import *


def main(model_type: str = 'xlm'):
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
    else:
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMModel.from_pretrained('xlm-mlm-100-1280')
    ner_model = XfmrNerModel(model_type, tokenizer, model)
    for project_type in ['fin', 'news']:
        rex_data_file_path = Path('data/processed/{}-{}-{}.csv'.format(project_type, model_type, 'rex'))
        rex_sentences = adm.read_project(Path('data/clean/project-{}/{}'.format('rex', project_type)))
        rex_labeled_sentences = label_sentences(rex_sentences, normalize_rex)
        df = process_rex_labeled_sentences(rex_labeled_sentences, ner_model)
        save_processed_dataset(df, rex_data_file_path)


if __name__ == "__main__":
    main()

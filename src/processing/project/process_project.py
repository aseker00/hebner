from pathlib import Path
from src.modeling.modeling_ner import create_token_ner_model, create_char_ner_model
from src.processing import processing_adm as adm
from src.processing.processing_utils import save_processed_dataset, load_processed_dataset, process_char_labeled_sentences, save_model_char_data_samples, process_token_labeled_sentences, save_model_token_data_samples
from src.processing.project.project_char_xfmr_dataset import label_char_sentences
from src.processing.project.project_token_xfmr_dataset import label_token_sentences


# # PER - person
# # ORG - organization
# # LOC - location
# # MISC - miscellaneous
# # TTL - title
# def normalize(label: str) -> str:
#     if label in ['MISC', 'TTL']:
#         return 'O'
#     return label


def main():
    gpe_label = 'gpe-loc'
    xfmr_model_type = 'xlm'
    ner_model_type = 'char'
    classifier_type = 'crf'
    project_sentences = {}
    if ner_model_type == 'char':
        spmrl_dataset_name = '{}-{}-{}-{}'.format('spmrl', ner_model_type, xfmr_model_type, gpe_label)
        spmrl_data_file_path = Path('data/processed/{}.csv'.format(spmrl_dataset_name))
        spmrl_df = load_processed_dataset(spmrl_data_file_path)
        spmrl_id2char = {a[0]: a[1] for a in spmrl_df[['char_id', 'char']].to_numpy()}
        spmrl_chars = [spmrl_id2char[char_id] for char_id in sorted(spmrl_id2char)]
        sent_chars = set()
        for project_type in ['news', 'fin']:
            annotated_sentences = adm.read_project_sentences(Path('data/clean/project/{}'.format(project_type)))
            labeled_sentences = label_char_sentences(annotated_sentences)
            project_sentences[project_type] = labeled_sentences
            sent_tokens = [sent.text[token_offset[0]:token_offset[1]] for sent in labeled_sentences for
                           token_offset in
                           sent.token_offsets]
            sent_chars.update({c for token in sent_tokens for c in list(token)})
        sorted_diff_chars = list(sorted([char for char in sent_chars if char not in spmrl_chars]))
        chars = spmrl_chars + sorted_diff_chars
        ft_model_path = Path("model/ft/cc.he.300.bin")
        ner_model = create_char_ner_model(chars, 0.3, xfmr_model_type, 0.3, classifier_type, ft_model_path)
        for project_type in ['news', 'fin']:
            labeled_sentences = project_sentences[project_type]
            dataset_name = '{}-{}-{}'.format(project_type, ner_model_type, xfmr_model_type)
            data_file_path = Path('data/processed/{}.csv'.format(dataset_name))
            sample_file_path = Path('data/processed/{}.pkl'.format(dataset_name))
            df = process_char_labeled_sentences(labeled_sentences, ner_model)
            save_processed_dataset(df, data_file_path)
            token_dataset_name = '{}-{}-{}'.format(project_type, 'token', xfmr_model_type)
            token_data_file_path = Path('data/processed/{}.csv'.format(token_dataset_name))
            token_df = load_processed_dataset(token_data_file_path)
            m = df.sent_idx.isin(token_df.sent_idx)
            save_model_char_data_samples(sample_file_path, labeled_sentences, token_df, df[m], ner_model)
    else:
        for project_type in ['news', 'fin']:
            annotated_sentences = adm.read_project_sentences(Path('data/clean/project/{}'.format(project_type)))
            token_labeled_sentences = label_token_sentences(annotated_sentences)
            project_sentences[project_type] = token_labeled_sentences
        ner_model = create_token_ner_model(xfmr_model_type, 0.3, classifier_type)
        for project_type in ['news', 'fin']:
            labeled_sentences = project_sentences[project_type]
            dataset_name = '{}-{}-{}'.format(project_type, ner_model_type, xfmr_model_type)
            data_file_path = Path('data/processed/{}.csv'.format(dataset_name))
            sample_file_path = Path('data/processed/{}.pkl'.format(dataset_name))
            df = process_token_labeled_sentences(labeled_sentences, ner_model)
            save_processed_dataset(df, data_file_path)
            save_model_token_data_samples(sample_file_path, labeled_sentences, df, ner_model)


if __name__ == "__main__":
    main()

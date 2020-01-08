from pathlib import Path
from src.modeling.modeling_ner import create_char_ner_model, create_token_ner_model
from src.processing import processing_conllu as conllu
from src.processing.processing_utils import process_char_labeled_sentences, process_token_labeled_sentences, save_processed_dataset, load_processed_dataset, save_model_char_data_samples, save_model_token_data_samples, spmrl_norm_labels_gpe_org, spmrl_norm_labels_gpe_loc
from src.processing.treebank.spmrl_char_xfmr_dataset import label_char_sentences
from src.processing.treebank.spmrl_token_xfmr_dataset import label_token_sentences


def main():
    gpe_label = 'gpe-org'
    xfmr_model_type = 'xlm'
    ner_model_type = 'char'
    classifier_type = 'crf'
    clean_data_file_path = Path('data/clean/treebank/spmrl-07.conllu')
    dataset_name = '{}-{}-{}-{}'.format('spmrl', ner_model_type, xfmr_model_type, gpe_label)
    sample_file_path = Path('data/processed/{}.pkl'.format(dataset_name))
    data_file_path = Path('data/processed/{}.csv'.format(dataset_name))
    lattice_sentences = conllu.read_conllu_sentences(clean_data_file_path, 'spmrl')
    norm_labels = spmrl_norm_labels_gpe_loc if gpe_label == 'gpe-loc' else spmrl_norm_labels_gpe_org
    if ner_model_type == 'char':
        labeled_sentences = label_char_sentences(lattice_sentences, norm_labels)
        sent_tokens = [sent.text[token_offset[0]:token_offset[1]] for sent in labeled_sentences for token_offset in sent.token_offsets]
        sent_chars = list(sorted({c for token in sent_tokens for c in list(token)}))
        ft_model_path = Path("model/ft/cc.he.300.bin")
        ner_model = create_char_ner_model(sent_chars, 0.3, xfmr_model_type, 0.3, classifier_type, ft_model_path)
        df = process_char_labeled_sentences(labeled_sentences, ner_model)
        token_dataset_name = '{}-{}-{}-{}'.format('spmrl', 'token', xfmr_model_type, gpe_label)
        token_data_file_path = Path('data/processed/{}.csv'.format(token_dataset_name))
        token_df = load_processed_dataset(token_data_file_path)
        save_model_char_data_samples(sample_file_path, labeled_sentences, token_df, df, ner_model)
    else:
        ner_model = create_token_ner_model(xfmr_model_type, 0.3, classifier_type)
        labeled_sentences = label_token_sentences(lattice_sentences, norm_labels)
        df = process_token_labeled_sentences(labeled_sentences, ner_model)
        save_model_token_data_samples(sample_file_path, labeled_sentences, df, ner_model)
    save_processed_dataset(df, data_file_path)


if __name__ == "__main__":
    main()

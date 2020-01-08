from src.processing.processing_utils import TokenLabeledSentence, normalize_project


def extract_token_label_offsets(sentence: tuple):
    _, token_offsets, _, _, entity_type_offsets, entity_token_offsets, token_entity_offsets = sentence
    label_offsets = {}
    for token_start_offset in sorted(token_offsets):
        if token_start_offset in label_offsets:
            continue
        label = 'O'
        if token_start_offset in token_entity_offsets:
            entity_start_offset = token_entity_offsets[token_start_offset]
            entity_type = normalize_project(entity_type_offsets[entity_start_offset])
            label = entity_type if entity_type == 'O' else 'B-' + entity_type
            label_offsets[token_start_offset] = label
            label = entity_type if entity_type == 'O' else 'I-' + entity_type
            for in_token_start_offset in entity_token_offsets[entity_start_offset][1:]:
                label_offsets[in_token_start_offset] = label
        else:
            label_offsets[token_start_offset] = label
    return label_offsets


def label_token_sentence(sent_id :int, sentence: tuple) -> TokenLabeledSentence:
    text, token_offsets, _, _, _, _, _ = sentence
    token_label_offsets = extract_token_label_offsets(sentence)
    token_labels = [token_label_offsets[token_start_offset] for token_start_offset in sorted(token_offsets)]
    return TokenLabeledSentence(sent_id, text, token_offsets, token_labels)


def label_token_sentences(annotated_sentences: list) -> list:
    return [label_token_sentence(i + 1, ann_sent) for i, ann_sent in enumerate(annotated_sentences)]

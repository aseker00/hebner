from src.processing.processing_utils import CharLabeledSentence, normalize_project


def label_char_sentence(sent_id :int, sentence: tuple) -> CharLabeledSentence:
    text, token_offsets, _, entity_offsets, entity_type_offsets, _, _ = sentence
    char_labels = ['O'] * len(text)
    for entity_start_offset in entity_offsets:
        entity_type = normalize_project(entity_type_offsets[entity_start_offset])
        if entity_type == 'O':
            continue
        label = 'B-' + entity_type
        char_labels[entity_start_offset] = label
        label = 'I-' + entity_type
        entity_end_offset = entity_offsets[entity_start_offset]
        char_labels[entity_start_offset + 1:entity_end_offset] = [label] * (entity_end_offset - entity_start_offset - 1)
    return CharLabeledSentence(sent_id, text, token_offsets, char_labels)


def label_char_sentences(annotated_sentences: list) -> list:
    return [label_char_sentence(i + 1, ann_sent) for i, ann_sent in enumerate(annotated_sentences)]

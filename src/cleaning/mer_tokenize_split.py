import json
from os import listdir
from os.path import join
from pathlib import Path
from src.processing import processing_adm as adm


def read_project(path: Path) -> list:
    annotations = []
    files_paths = [join(str(path), f) for f in listdir(str(path))]
    print("{} files in project {}".format(len(files_paths), str(path)))
    for file_path in [p for p in files_paths if p[-4:] == 'json']:
        with open(file_path, 'r') as f:
            annotated_json = json.load(f)
            annotations.append(annotated_json)
    return annotations


# def remove_x_entities(annotations: dict) -> dict:
#     token_offsets, token_tags, token_items = adm.get_annotated_token_offsets(annotations)
#     entity_offsets, entity_types, entity_items = adm.get_annotated_entity_offsets(annotations)
#     entity_token_offsets = adm.get_entity_token_offsets(token_offsets, entity_offsets)
#     no_x_entity_items = []
#     for entity_start_offset in entity_token_offsets:
#         x_entity = False
#         for token_start_offset in entity_token_offsets[entity_start_offset]:
#             token_item = token_items[token_start_offset]
#             analysis = token_item['analyses'][0]
#             tag = analysis['partOfSpeech']
#             if tag == 'X':
#                 x_entity = True
#                 break
#         if not x_entity:
#             entity_item = entity_items[entity_start_offset]
#             no_x_entity_items.append(entity_item)
#     clean_annotations = {'data': annotations['data']}
#     clean_annotations['attributes'] = {'entities': {'type': 'list', 'itemType': 'entities', 'items': no_x_entity_items}}
#     clean_annotations['attributes']['token'] = annotations['attributes']['token']
#     return clean_annotations
def save_sentences(raw_annotations, clean_project_path: Path):
    annotated_sentences = []
    for annotation in raw_annotations:
        tokenized_annotation = adm.tokenize(annotation)
        annotated_sentences.extend(adm.split(tokenized_annotation))
    for sent_index, annotated_sent in enumerate(annotated_sentences):
        # clean_annotation = remove_x_entities(annotated_sent)
        with open('{}/{}.adm.json'.format(str(clean_project_path), sent_index + 1), 'w') as f:
            json.dump(annotated_sent, f)


for data_type in ['train', 'eval']:
    raw_annotations = read_project(Path(f'data/raw/mer/MER-2020-02-11/{data_type}'))
    save_sentences(raw_annotations, Path(f'data/clean/mer/MER-2020-02-11/{data_type}'))

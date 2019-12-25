import json
from collections import defaultdict
from copy import deepcopy
from os import listdir
from os.path import join
from pathlib import Path

from rosette.api import API, DocumentParameters, RosetteException
import warnings
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


api = API(user_key="50df9d3641edb5ec46924693378ae13c")
api.set_url_parameter("output", "rosette")
# result = api.ping()
# print("/ping: ", result)


def get_tokens(data: str):
    params = DocumentParameters()
    params["content"] = data
    params["language"] = 'heb'
    try:
        return api.tokens(params)
    except RosetteException as exception:
        print(exception)


def get_entities(data: str):
    params = DocumentParameters()
    params["content"] = data
    params["language"] = 'heb'
    try:
        return api.entities(params)
    except RosetteException as exception:
        print(exception)


def get_annotated_sentences(annotations: dict) -> dict:
    sentence_offsets = {}
    for item in annotations['attributes']['sentence']['items']:
        start_offset = int(item['startOffset'])
        end_offset = int(item['endOffset'])
        sentence_offsets[start_offset] = end_offset
    return sentence_offsets


def get_annotated_tokens(annotations: dict) -> (dict, dict, dict):
    token_offsets = {}
    token_tags = {}
    token_items = {}
    for item in annotations['attributes']['token']['items']:
        start_offset = int(item['startOffset'])
        end_offset = int(item['endOffset'])
        tag = item['analyses'][0]['partOfSpeech'] if 'partOfSpeech' in item['analyses'][0] else None
        token_offsets[start_offset] = end_offset
        token_tags[start_offset] = tag
        token_items[start_offset] = item
    return token_offsets, token_tags, token_items


def get_annotated_entities(annotations: dict) -> (dict, dict, dict):
    entity_offsets = {}
    entity_types = {}
    entity_items = {}
    for item in annotations['attributes']['entities']['items']:
        start_offset = int(item['mentions'][0]['startOffset'])
        end_offset = int(item['mentions'][0]['endOffset'])
        entity_type = item['type']
        entity_offsets[start_offset] = end_offset
        entity_types[start_offset] = entity_type
        entity_items[start_offset] = item
    return entity_offsets, entity_types, entity_items


def get_token_entities(token_offsets: dict, entity_offsets: dict) -> (dict, dict):
    sorted_token_start_offsets = sorted(token_offsets)
    sorted_entity_start_offsets = sorted(entity_offsets)
    sorted_token_start_offset_index = 0
    entity_tokens = defaultdict(list)
    for entity_start_offset in sorted_entity_start_offsets:
        entity_end_offset = entity_offsets[entity_start_offset]
        token_start_offset = sorted_token_start_offsets[sorted_token_start_offset_index]
        token_end_offset = token_offsets[token_start_offset]
        while token_end_offset < entity_start_offset:
            sorted_token_start_offset_index += 1
            if sorted_token_start_offset_index == len(sorted_token_start_offsets):
                break
            token_start_offset = sorted_token_start_offsets[sorted_token_start_offset_index]
            token_end_offset = token_offsets[token_start_offset]
        while token_end_offset <= entity_end_offset:
            entity_tokens[entity_start_offset].append(sorted_token_start_offsets[sorted_token_start_offset_index])
            sorted_token_start_offset_index += 1
            if sorted_token_start_offset_index == len(sorted_token_start_offsets):
                break
            token_start_offset = sorted_token_start_offsets[sorted_token_start_offset_index]
            token_end_offset = token_offsets[token_start_offset]
    token_entity_offsets = {}
    for entity_start_offset in entity_tokens:
        for token_start_offset in entity_tokens[entity_start_offset]:
            token_entity_offsets[token_start_offset] = entity_start_offset
    return entity_tokens, token_entity_offsets


def tokenize(annotations: dict) -> dict:
    annotations_copy = deepcopy(annotations)
    tokenized_annotations = get_tokens(annotations['data'])
    annotations_copy['attributes']['sentence'] = tokenized_annotations['attributes']['sentence']
    annotations_copy['attributes']['token'] = tokenized_annotations['attributes']['token']
    return annotations_copy


def split(annotations) -> list:
    annotated_sentences = get_annotated_sentences(annotations)
    annotated_tokens, _, token_items = get_annotated_tokens(annotations)
    annotated_entities, _, entity_items = get_annotated_entities(annotations)
    sentences = []
    for sentence_start_offset in annotated_sentences:
        sentence_end_offset = annotated_sentences[sentence_start_offset]
        sentence_text = annotations['data'][sentence_start_offset:sentence_end_offset].replace("\n", " ").replace("\t", " ")
        sentence_annotations = {'data': sentence_text}
        sentence_entity_items = {}
        sentence_entity_offsets = {}
        for entity_start_offset in sorted([offset for offset in annotated_entities if sentence_start_offset <= offset < sentence_end_offset]):
            entity_item = entity_items[entity_start_offset]
            entity_end_offset = annotated_entities[entity_start_offset]
            sentence_entity_items[entity_start_offset - sentence_start_offset] = entity_item
            sentence_entity_offsets[entity_start_offset - sentence_start_offset] = entity_end_offset - sentence_start_offset
        sentence_token_items = {}
        sentence_token_offsets = {}
        for token_start_offset in [offset for offset in annotated_tokens if sentence_start_offset <= offset < sentence_end_offset]:
            token_item = token_items[token_start_offset]
            token_end_offset = annotated_tokens[token_start_offset]
            sentence_token_items[token_start_offset - sentence_start_offset] = token_item
            sentence_token_offsets[token_start_offset - sentence_start_offset] = token_end_offset - sentence_start_offset
        items = []
        for sentence_entity_start_offset in sentence_entity_items:
            sentence_entity_end_offset = sentence_entity_offsets[sentence_entity_start_offset]
            sentence_entity_item = sentence_entity_items[sentence_entity_start_offset]
            item = deepcopy(sentence_entity_item)
            item['mentions'][0]['startOffset'] = sentence_entity_start_offset
            item['mentions'][0]['endOffset'] = sentence_entity_end_offset
            items.append(item)
        sentence_annotations['attributes'] = {'entities': {'type': 'list', 'itemType': 'entities', 'items': items}}
        items = []
        for sentence_token_start_offset in sentence_token_items:
            sentence_token_end_offset = sentence_token_offsets[sentence_token_start_offset]
            sentence_token_item = sentence_token_items[sentence_token_start_offset]
            item = deepcopy(sentence_token_item)
            item['startOffset'] = sentence_token_start_offset
            item['endOffset'] = sentence_token_end_offset
            items.append(item)
        sentence_annotations['attributes']['token'] = {'type': 'list', 'itemType': 'token', 'items': items}
        sentences.append(sentence_annotations)
    return sentences


def extract_entity_mentions(annotations: dict) -> (list, list):
    mentions = []
    types = []
    entity_offsets, entity_types, _ = get_annotated_entities(annotations)
    for entity_start_offset in entity_offsets:
        entity_end_offset = entity_offsets[entity_start_offset]
        entity_type = entity_types[entity_start_offset]
        entity_mention = annotations['data'][entity_start_offset:entity_end_offset]
        types.append(entity_type)
        mentions.append(entity_mention)
    return mentions, types


def read_project(project_dir_path: Path) -> list:
    sentences = []
    files_paths = [join(str(project_dir_path), f) for f in listdir(str(project_dir_path))]
    for file_path in files_paths:
        with open(file_path, 'r') as f:
            annotated_sentence = json.load(f)
            text = annotated_sentence['data']
            tokens, tags, _ = get_annotated_tokens(annotated_sentence)
            entities, entity_types, _ = get_annotated_entities(annotated_sentence)
            entity_tokens, token_entities = get_token_entities(tokens, entities)
            sentences.append((text, tokens, tags, entities, entity_types, entity_tokens, token_entities))
    return sentences


def annotate_entities(dir_path: Path) -> list:
    for file_name in listdir(str(dir_path)):
        file_path = join(str(dir_path), file_name)
        with open(file_path, 'r') as f:
            input_annotation = json.load(f)
            text = input_annotation['data']
            annotated_output = get_entities(text)
            yield file_name, annotated_output

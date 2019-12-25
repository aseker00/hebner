from pathlib import Path
from src.processing import processing_adm as adm
import json

for project_type in ['fin', 'news']:
    output_dir_path = Path('data/clean/project-rex/{}'.format(project_type))
    annotated_sentences = adm.annotate_entities(Path('data/clean/project/{}'.format(project_type)))
    for file_name, annotated_sent in annotated_sentences:
        with open('{}/{}'.format(output_dir_path, file_name), 'w') as f:
            json.dump(annotated_sent, f)

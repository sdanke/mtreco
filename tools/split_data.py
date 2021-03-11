import os
import sys
import argparse

from glob import glob
from tqdm import tqdm
from random import random
sys.path.insert(0, f'{sys.path[0]}/..')
from ocr.utils.io import load_json_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split train & val data"
    )

    parser.add_argument(
        "--base_dir", type=str, help="The generated data directory", default="data/generated"
    )
    args = parser.parse_args()
    BASE_DIR = args.base_dir
    IMG_DIR = f'{BASE_DIR}/imgs'
    LABEL_DIR = f'{BASE_DIR}/labels'
    label_files = glob(f'{LABEL_DIR}\\*.*')

    full_id_writer = open(f'{BASE_DIR}/dataset_ids.txt', 'w', encoding='utf-8')
    train_id_writer = open(f'{BASE_DIR}/train_ids.txt', 'w', encoding='utf-8')
    val_id_writer = open(f'{BASE_DIR}/val_ids.txt', 'w', encoding='utf-8')
    for label_file in tqdm(label_files):
        img_id = label_file.split('\\')[-1].split('.')[0]
        img_path = f'{IMG_DIR}/{img_id}.jpg'
        if not os.path.exists(img_path):
            continue
        label = load_json_file(label_file)
        if label is None:
            continue
        assert True
        text = label['text']
        bboxes = label['bboxes']
        txt_line = f'{img_id}[SEP]{text}[SEP]{bboxes}\n'
        id_line = f'{img_id}\n'
        full_id_writer.write(id_line)
        if random() < 0.01:
            val_id_writer.write(id_line)
        else:

            train_id_writer.write(id_line)

    full_id_writer.close()
    train_id_writer.close()
    val_id_writer.close()

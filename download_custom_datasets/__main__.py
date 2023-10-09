import os
import logging

logging.basicConfig()
log = logging.getLogger("vectordb_bench")
log.setLevel(logging.INFO)

DATASETS = [
    { 'name': 'sift', 'folder_name': 'sift_small_1m', 'url': 'https://storage.googleapis.com/lanterndata/vectordb_bench', 'files': ['train.parquet', 'test.parquet', 'neighbors.parquet', 'neighbors_head_1p.parquet', 'neighbors_tail_1p.parquet'] }
]

DATASET_DIR = '/tmp/vectordb_bench/dataset'

def main():
    for dataset in DATASETS:
        dataset_folder = f'{DATASET_DIR}/{dataset["name"]}/{dataset["folder_name"]}'
        if os.path.exists(dataset_folder):
            log.info(f'Dataset {dataset["name"]} already downloaded')
            continue

        os.system(f'mkdir -p {dataset_folder}')

        for file_name in dataset['files']:
            os.system(f'wget {dataset["url"]}/{dataset["folder_name"]}/{file_name} -P {dataset_folder}')

        log.info(f'Dataset {dataset["name"]} successfully downloaded')

if __name__ == "__main__":
    main()

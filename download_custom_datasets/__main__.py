import os
import logging

logging.basicConfig()
log = logging.getLogger("vectordb_bench")
log.setLevel(logging.INFO)

DATASETS = [
    { 'name': 'sift', 'folder_name': 'sift_small_1m', 'url': 'https://storage.googleapis.com/lanterndata/vectordb_bench', 'files': ['train.parquet', 'test.parquet', 'neighbors.parquet', 'neighbors_head_1p.parquet', 'neighbors_tail_1p.parquet'] },
    { 'name': 'cohere', 'folder_name': 'cohere_xlarge_35m', 'url': 'https://storage.googleapis.com/lanterndata/vectordb_bench', 'files': ['test.parquet', 'neighbors.parquet'] },
    { 'name': 'cohere', 'folder_name': 'cohere_xlarge_35m', 'url': 'https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-en-embeddings/parquet/default/train', 'files': [{ 'name': 'shuffle_train', 'range': range(253) }] }
]

DATASET_DIR = '/tmp/vectordb_bench/dataset'

FILTER = os.getenv('FILTER')

def main():
    for dataset in DATASETS:
        if FILTER and not dataset['name'].startswith(FILTER):
            continue
        dataset_folder = f'{DATASET_DIR}/{dataset["name"]}/{dataset["folder_name"]}'

        os.system(f'mkdir -p {dataset_folder}')

        for file_name in dataset['files']:
            if type(file_name) == dict:
                if not file_name.get('range'):
                    log.error(f'Please specify range of parts for {file_name.get("name")} to download')
                    break

                split_cnt = str(len(file_name.get('range'))).zfill(2)
                for part in file_name.get('range'):
                    paded_name = str(part).zfill(2)
                    filedest = f'{dataset_folder}/{file_name["name"]}-{paded_name}-of-{split_cnt}.parquet'
                    os.system(f'wget {dataset["url"]}/{paded_name}.parquet -O {filedest}')
                    log.info(f'Dataset {dataset["name"]} part {paded_name} successfully downloaded')

                continue
                    
            if os.path.exists(f'{dataset_folder}/{file_name}'):
                log.info(f'Dataset file {dataset_folder}/{file_name} already downloaded')
                continue
            os.system(f'wget -q {dataset["url"]}/{dataset["folder_name"]}/{file_name} -P {dataset_folder}')

        log.info(f'Dataset {dataset["name"]} successfully downloaded')

if __name__ == "__main__":
    main()


from pathlib import Path
BASE_DIR = Path('pybert')
config = {
    'raw_data_path': BASE_DIR/r'dataset/emse.csv',
    # 'test_path': BASE_DIR/r'',
    'aug_data_path': BASE_DIR/r'dataset/TTA_ALL .csv',
    # 'aug_test_path': BASE_DIR/r'',

    'data_dir': BASE_DIR/r"dataset",
    'log_dir': BASE_DIR/r'output/log',
    'writer_dir': BASE_DIR/r"output/TSboard",
    'figure_dir': BASE_DIR/r"output/figure",
    'checkpoint_dir': BASE_DIR/r"output/checkpoints",
    'cache_dir': BASE_DIR /r'model/',
    'result': BASE_DIR /r"output/result",

    'bert_vocab_path': BASE_DIR/r'pretrain/bert/base-uncased/vocab.txt',
    'bert_config_file': BASE_DIR/r'pretrain/bert/base-uncased/config.json',
    'bert_model_dir': BASE_DIR/r'pretrain/bert/base-uncased',

}

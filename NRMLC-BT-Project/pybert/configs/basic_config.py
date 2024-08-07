
from pathlib import Path
BASE_DIR = Path('pybert')
config = {
    'raw_data_path': BASE_DIR/r'dataset/review_origin_train.csv',
    'test_path': BASE_DIR/r'dataset/review_origin_test.csv',
    'aug_data_path': BASE_DIR/r'dataset/TTA_Per.csv',
    'aug_test_path': BASE_DIR/r'dataset/TTA_Test.csv',

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

    'xlnet_vocab_path': BASE_DIR / 'pretrain/xlnet/base-cased/spiece.model',
    'xlnet_config_file': BASE_DIR / 'pretrain/xlnet/base-cased/config.json',
    'xlnet_model_dir': BASE_DIR / 'pretrain/xlnet/base-cased',

    'albert_vocab_path': BASE_DIR / 'pretrain/albert/albert-base/30k-clean.model',
    'albert_config_file': BASE_DIR / 'pretrain/albert/albert-base/config.json',
    'albert_model_dir': BASE_DIR / 'pretrain/albert/albert-base'


}

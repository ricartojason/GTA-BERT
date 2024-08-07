import random
import pandas as pd
from tqdm import tqdm
from ..common.tools import save_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar

class TaskData(object):
    def __init__(self):
        pass
    # x表示特征数据，y表示标签数据，valid_size表示验证集的比例，shuffle: 是否随机打乱数据，
    # stratify: 是否根据标签 y 来分层抽样，以保持训练集和验证集中类别的比例与整个数据集相同。
    def train_val_split(self,X, y,valid_size,stratify=False,shuffle=True,save = True,
                        seed = None,data_name = None,data_dir = None):
        # 创建进度条
        pbar = ProgressBar(n_total=len(X),desc='bucket')
        # 记录日志，提示正在将原始数据划分为训练集和验证集
        logger.info('split raw data into train and valid')
        if stratify:
            # 计算类别的数量
            # set设置集合，里面不能包含重复的元素，接收一个list作为参数
            num_classes = len(list(set(y)))
            train, valid = [], []
            # 初始化每个类别的桶
            bucket = [[] for _ in range(num_classes)]
            # 遍历原始数据，按照类别划分到不同的桶中
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                bucket[int(data_y)].append((data_x, data_y))
                pbar(step=step)
            # 删除原始数据
            del X, y
            # 遍历每个桶，按照验证集大小划分数据
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                test_size = int(N * valid_size)
                if shuffle:
                    random.seed(seed)
                    random.shuffle(bt)
                # 将数据划分到验证集中
                valid.extend(bt[:test_size])
                # 将剩余数据划分到训练集中
                train.extend(bt[test_size:])
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        # 如果不需要按照类别划分，直接遍历原始数据
        else:
            data = []
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                data.append((data_x, data_y))
                pbar(step=step)
            del X, y
            # 计算数据的总数量
            N = len(data)
            # 计算验证集的大小
            test_size = int(N * valid_size)
            if shuffle:
                random.seed(seed)
                random.shuffle(data)
            valid = data[:test_size]
            train = data[test_size:]
            # 混洗train数据集
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        if save:
            # 定义训练集和验证集的保存路径
            train_path = data_dir / f"{data_name}.train.pkl"
            valid_path = data_dir / f"{data_name}.valid.pkl"
            save_pickle(data=train,file_path=train_path)
            save_pickle(data = valid,file_path=valid_path)
        # 返回训练集和验证集
        return train, valid

    def read_data(self,raw_data_path,aug_data_path,preprocessor = None,is_train=True,is_augament=True):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        targets, sentences = [], []
        data = pd.read_csv(raw_data_path)
        for row in data.values:
            if is_train:
                #取原始数据里第2列到第6列，即非功能需求标签[1,0,0,0,0]
                target = row[2:]
            else:
                target = row[2:]
            sentence = str(row[1])
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentence)
        if is_augament:
            TTA_data = pd.read_csv(aug_data_path)
            for row1 in TTA_data.values:
                aug_target = row1[2:]
                aug_sentence = str(row1[1])
                if preprocessor:
                    aug_sentence = preprocessor(aug_sentence)
                if sentence:
                    targets.append(aug_target)
                    sentences.append(aug_sentence)
        return targets,sentences

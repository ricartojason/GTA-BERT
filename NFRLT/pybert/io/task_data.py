import random
import pandas as pd
from tqdm import tqdm
from ..common.tools import save_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar

class TaskData(object):
    def __init__(self):
        pass
    # x表示特征数据，y表示标签数据，train_size表示训练集的比例，shuffle: 是否随机打乱数据，
    # stratify: 是否根据标签 y 来分层抽样，以保持训练集和验证集中类别的比例与整个数据集相同。
    def train_val_split(self,X, y,train_size,stratify=False,shuffle=True,save = True,
                        seed = None,data_name = None,data_dir = None):
        # 创建进度条
        pbar = ProgressBar(n_total=len(X),desc='bucket')
        # 记录日志，提示正在将原始数据划分为训练集和验证集
        test_size = 0.1
        logger.info('split raw data into train and valid and test')
        if stratify:
            # 计算类别的数量
            # set设置集合，里面不能包含重复的元素，接收一个list作为参数
            num_classes = len(list(set(y)))
            train, valid, test = [], [], [] 
            # 初始化每个类别的桶
            bucket = [[] for _ in range(num_classes)]
            # 遍历原始数据，按照类别划分到不同的桶中
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                bucket[int(data_y)].append((data_x, data_y))
                pbar(step=step)
            # 删除原始数据
            del X, y
            # 遍历每个桶，按照训练集大小划分数据
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                train_size = int(N * train_size)
                test_size = int(N * test_size)
                if shuffle:
                    random.seed(seed)
                    random.shuffle(bt)
                # 将数据划分到训练集中
                train.extend(bt[:train_size])
                # 将剩余数据划分到验证集中
                valid.extend(bt[train_size:train_size+test_size])
                # 将剩余数据划分到测试集中
                test.extend(bt[train_size+test_size:])
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
            # 计算训练集的大小
            train_size = int(N * train_size)
            test_size = int(N * test_size)
            if shuffle:
                random.seed(seed)
                random.shuffle(data)
            train = data[:train_size]
            valid = data[train_size:train_size+test_size]
            test = data[train_size+test_size:]
            # 混洗train数据集
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        if save:
            # 定义训练集和验证集的保存路径
            train_path = data_dir / f"{data_name}.train.pkl"
            valid_path = data_dir / f"{data_name}.valid.pkl"
            test_path = data_dir / f"{data_name}.test.pkl"
            save_pickle(data= train, file_path=train_path)
            save_pickle(data = valid, file_path=valid_path)
            save_pickle(data = test, file_path=test_path)
        # 返回训练集和验证集
        return train_path, valid_path, test_path 

    def read_data(self, raw_data_path, aug_data_path, aug_test_path, preprocessor = None,is_train=True,is_augament=True):
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
            aug_data = pd.read_csv(aug_test_path)
            combined_data = pd.concat([TTA_data, aug_data], ignore_index=True)
            combined_data = combined_data.drop_duplicates()
            for row1 in combined_data.values:
                aug_target = row1[2:]
                aug_sentence = str(row1[1])
                if preprocessor:
                    aug_sentence = preprocessor(aug_sentence)
                if sentence:
                    targets.append(aug_target)
                    sentences.append(aug_sentence)
        return targets,sentences

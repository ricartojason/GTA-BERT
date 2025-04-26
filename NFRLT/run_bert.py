import torch
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss, BCEWithLoss, MultiLabelCrossEntropy
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.bert_for_multi_label import BertForMultiLable
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport, Precision, Recall, HammingScore, HammingLoss, \
    F1Score, ClassReport, Jaccard, Accuracy
from pybert.callback.optimizater.adamw import AdamW
from pybert.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler
from torchsummary import summary
from pybert.common.tools import load_pickle

warnings.filterwarnings("ignore")

# %%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cuda:1"
device = torch.device(dev)


def run_train(args, train, valid, log_name):
    # --------- data
    # Bert对象，用于处理相关数据，参数是词汇表路径和是否将文本转换为小写
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    #这个部分可以作数据增强(data_augmentation_func=augment_data)
    train_examples = processor.create_examples(lines=train,
                                               example_type='train',
                                               cached_examples_file=config[
                                                                        'data_dir'] / f"cached_train_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                                        'data_dir'] / "cached_train_features_{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))
    # 将包含特征的列表features使用列表推导式转换为PyTorch的TensorDataset对象
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    # collate_fn: 这是一个函数，用于在创建批次时将多个样本合并成一个批次的张量。当数据集中的样本具有不同的长度或形状时，collate_fn
    # 负责处理这些差异，例如通过填充（padding）来确保所有样本具有相同的长度。
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    valid_examples = processor.create_examples(lines=valid,
                                               example_type='valid',
                                               cached_examples_file=config[
                                                                        'data_dir'] / f"cached_valid_examples_{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=config[
                                                                        'data_dir'] / "cached_valid_features_{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
                                  collate_fn=collate_fn)
    print('数据处理完成了吗')

    # ------- model
    logger.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BertForMultiLable.from_pretrained(args.resume_path, num_labels=len(label_list)).to(device)
    else:
        model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list)).to(device)
    # # 打印模型结构
    # print(model)

    # t_total表示在整个训练过程中（包括所有周期和梯度累积步骤）需要遍历的总批次数
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    # 这段代码的目的是获取model中的所有参数及其名称，并将它们存储在一个列表中
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    # 如偏置项和层归一化的权重通常不需要权重衰减
    optimizer_grouped_parameters = [
        # 检查 n（参数名）中是否包含no_decay列表中的任何片段。如果包含，any函数将返回True，否则返回False，不包含就在第一个字典，p是参数的值
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    # 如果args.fp16为True，则代码尝试使用NVIDIA的Apex库来初始化模型和优化器，以进行FP16（16
    # 位浮点数）训练。FP16训练可以加速训练过程并减少内存使用，但可能需要特殊的硬件和软件支持
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    print('模型构建完成了吗')
    # ---- callbacks
    # 回调是一种常用的机制，允许用户在训练的不同阶段（如每个epoch的开始或结束）执行自定义的操作。
    logger.info("initializing callbacks")
    # 创建了一个TrainingMonitor类的实例。这个类可能是用来监控和记录训练过程中的各种指标，如损失、准确率等，并可能将这些指标可视化。
    # mode = args.mode：指定了保存模型权重的模式（例如，'min'或'max'）。
    # 这通常与monitor参数一起使用，来决定何时保存模型权重。
    # monitor = args.monitor：指定了用于监控的指标，
    # 如损失或准确率。当这个指标达到某个阈值时，
    # ModelCheckpoint可能会保存模型的权重
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch, log_name=log_name)
    # 在训练过程中保存模型权重的。这里传入了以下参数
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'] / log_name, mode=args.mode,
                                       monitor=args.monitor, arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(args=args, model=model, logger=logger, criterion=BCEWithLogLoss(), optimizer=optimizer,
                      scheduler=scheduler, early_stopping=None, training_monitor=train_monitor,
                      model_checkpoint=model_checkpoint,
                      batch_metrics=[
                          Recall(task_type='binary', average='macro'),  # 添加召回率指标
                          Jaccard(average='macro'),
                          F1Score(thresh=0.5, normalizate=True, task_type='binary', average='macro',
                                  search_thresh=False)  # 添加 F1 分数指标
                      ],
                      epoch_metrics=[
                          Precision(task_type='binary', average='macro'),  # 同样添加精确度指标
                          Recall(task_type='binary', average='macro'),  # 同样添加召回率指标
                          HammingScore(),  # 同样添加汉明分数指标
                          HammingLoss(),  # 同样添加汉明损失指标
                          F1Score(thresh=0.5, normalizate=True, task_type='binary', average='macro',
                                  search_thresh=False),  # 添加 F1 分数指标
                          MultiLabelReport(id2label=id2label, average='macro', logger=logger),
                          Jaccard(average='macro'),  # 同样添加 jaccard 指标
                          ClassReport()
                      ])
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)



# %%
def run_test(args, test):
    from pybert.test.predictor import Predictor
    import torch
    
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}

    # 检查test是否为路径对象，并加载数据
    if isinstance(test, Path):
        logger.info(f"Loading test data from file: {test}")
        test_data = load_pickle(test)
    else:
        test_data = test

    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file=config[
                                                                       'data_dir'] / f"cached_test_examples_{args.arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=args.eval_max_seq_len,
                                              cached_features_file=config[
                                                                       'data_dir'] / "cached_test_features_{}_{}".format(
                                                  args.eval_max_seq_len, args.arch
                                              ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size,
                                 collate_fn=collate_fn)
    
    # 修改checkpoint路径
    checkpoint_path = r"C:\Users\wyf\Desktop\research\GPT-BERT\GPT-BERT-final\pybert\output\checkpoints\bert+smooth\bert+smooth-2025-04-16-19_05_04-1e-05-16-6"
    logger.info(f"加载模型: {checkpoint_path}")
    try:
        model = BertForMultiLable.from_pretrained(checkpoint_path, num_labels=len(label_list))
        logger.info("模型加载成功")
        # 打印模型结构简要信息
        logger.info(f"模型结构: {', '.join([name for name, _ in model.named_children()])}")
        model = model.to(device)
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e

    # ----------- predicting
    logger.info('开始预测...')
    
    # 准备所有要评估的指标
    test_metrics = [
        F1Score(thresh=0.5, normalizate=True, task_type='binary', average='macro', search_thresh=False),
        Precision(task_type='binary', average='macro'),
        Recall(task_type='binary', average='macro'),
        HammingScore(),
        HammingLoss(),
        Jaccard(average='macro'),
        MultiLabelReport(id2label=id2label, average='macro', logger=logger),
        ClassReport(logger=logger)  # 传递logger给ClassReport
    ]
    
    # 创建预测器
    predictor = Predictor(model=model, logger=logger, n_gpu=args.n_gpu, test_metrics=test_metrics)
    
    # 进行预测和评估
    result = predictor.predict(data=test_dataloader)
    
    # 打印结果摘要
    logger.info("\n测试结果摘要:")
    for metric_name, value in result.items():
        if isinstance(value, (int, float)):
            logger.info(f"{metric_name}: {value:.4f}")
    
    return result


# %%
def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert+smooth', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='emse', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--train_size", default=0.8, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument("--train_max_seq_len", default=128, type=int)
    parser.add_argument("--eval_max_seq_len", default=128, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=1e-05, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()

    # 调用了init_logger函数来初始化日志记录器，将日志记录到指定的文件中。文件名由args.arch和当前时间戳组成。
    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    log_name = f'{args.arch}-{timestamp}-{args.learning_rate}-{args.train_batch_size}-{args.epochs}'
    log_file_path = Path(config['log_dir']) / f'{log_name}.log'
    
    init_logger(log_file_path)
    # 设置检查点目录:
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_data:
        from pybert.io.task_data import TaskData
        data = TaskData()
        targets, sentences = data.read_data(raw_data_path=config['raw_data_path'],
                                            aug_data_path=config['aug_data_path'],
                                            aug_test_path=config['aug_test_path'],
                                            preprocessor=EnglishPreProcessor(),
                                            is_train=True,
                                            is_augament=False)
        data.train_val_split(X=sentences, y=targets, shuffle=True, stratify=False,
                             train_size=args.train_size, data_dir=config['data_dir'],
                             data_name=args.data_name)
    if args.do_train:
        from pybert.io.task_data import TaskData
        data = TaskData()
        import numpy as np
        # 获取标签列表
        processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
        label_list = processor.get_labels()
        num_labels = len(label_list)
        
        targets, sentences = data.read_data(raw_data_path=config['raw_data_path'],
                                aug_data_path=config['aug_data_path'],
                                aug_test_path=config['aug_test_path'],
                                preprocessor=EnglishPreProcessor(),
                                is_train=True,
                                is_augament=False)
        
        

        # 计算标签权重
        # label_counts = [0] * num_labels  # 使用实际的标签数量
        # total_samples = len(targets)
        # for target in targets:
        #     for i in range(num_labels):  # 遍历每个标签
        #         label_counts[i] += target[i]

        # label_weights = [(total_samples - count) / count for count in label_counts]
        # label_weights = [ np.sqrt(w) for w in label_weights]
        
        # 计算每个标签的权重（少数类获得更高权重）
        # 使用逆频率作为权重，但不进行归一化
        # label_weights = [total_samples / (count + 1) for count in label_counts]
        # 对权重进行缩放，使其在合理范围内
        # max_weight = max(label_weights)
        # label_weights = [w / max_weight for w in label_weights]

        # 将标签权重添加到args中
        args.label_weights = label_weights
        print(f"Label weights calculated: {label_weights}")

        # 打印调试信息
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Label counts: {label_counts}")

        # 检查是否已经有数据，如果没有则加载
        if 'train' not in locals():
            # 从保存的文件加载训练和验证数据
            from pybert.io.task_data import TaskData
            processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
            
            # 加载之前保存的训练和验证数据
            train = processor.get_train(config['data_dir'] / f"{args.data_name}.train.pkl")
            valid = processor.get_dev(config['data_dir'] / f"{args.data_name}.valid.pkl")
            
            # 如果找不到保存的数据文件，提示用户需要先运行--do_data
            if not train or not valid:
                logger.error("未找到训练和验证数据，请先运行 --do_data 参数来准备数据")
                return
            
        run_train(args, train, valid, log_name)

    if args.do_test:
        # 从保存的文件加载测试数据
        from pybert.io.task_data import TaskData
        processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
        
        # 加载之前保存的测试数据
        logger.info(f"加载测试数据: {config['data_dir'] / f'{args.data_name}.test.pkl'}")
        test = processor.get_test(config['data_dir'] / f"{args.data_name}.test.pkl")
        
        # 如果找不到保存的数据文件，提示用户
        if not test:
            logger.error(f"未找到测试数据文件: {config['data_dir'] / f'{args.data_name}.test.pkl'}")
            return
        
        run_test(args, test)


if __name__ == '__main__':
    main()

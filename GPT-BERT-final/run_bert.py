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

warnings.filterwarnings("ignore")

# %%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cuda:1"
device = torch.device(dev)


def run_train(args):
    # --------- data
    # Bert对象，用于处理相关数据，参数是词汇表路径和是否将文本转换为小写
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    # 指定路径加载训练数据，它将args.data_name的值与字符串.train.pkl拼接起来，形成完整的文件名。例如，如果args.data_name是"my_data"，那么完整的文件名就是"my_data.train.pkl"。
    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}.train.pkl")
    #这个部分可以作数据增强(data_augmentation_func=augment_data)
    train_examples = processor.create_examples(lines=train_data,
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

    valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}.valid.pkl")
    valid_examples = processor.create_examples(lines=valid_data,
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
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    # 在训练过程中保存模型权重的。这里传入了以下参数
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'], mode=args.mode,
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
                          AccuracyThresh(search_thresh=True),
                          Precision(task_type='binary', average='macro',search_thresh=True),  # 添加精确度指标
                          Recall(task_type='binary', average='macro',search_thresh=True),  # 添加召回率指标
                          # 如果你想要计算汉明损失和分数，你需要确保它们是针对多标签分类问题
                          HammingScore(search_thresh=True),  # 添加汉明分数指标
                          HammingLoss(search_thresh=True),  # 添加汉明损失指标
                          F1Score(thresh=0.5, normalizate=True, task_type='binary', average='macro',
                                  search_thresh=True)  # 添加 F1 分数指标

                      ],
                      epoch_metrics=[
                          AccuracyThresh(search_thresh=True),
                          # Accuracy(topK=4),
                          AUC(task_type='binary', average='macro',search_thresh = True),
                          Precision(task_type='binary', average='macro',search_thresh=True),  # 同样添加精确度指标
                          Recall(task_type='binary', average='macro',search_thresh=True),  # 同样添加召回率指标
                          HammingScore(search_thresh=True),  # 同样添加汉明分数指标
                          HammingLoss(search_thresh=True),  # 同样添加汉明损失指标
                          F1Score(thresh=0.5, normalizate=True, task_type='binary', average='macro',
                                  search_thresh=True),  # 同样添加 F1 分数指标
                          MultiLabelReport(id2label=id2label, average='macro', logger=logger),
                          Jaccard(average='macro',search_thresh=True),  # 同样添加 jaccard 指标
                          ClassReport()
                      ])
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)



# %%
def run_test(args):
    from pybert.io.task_data import TaskData
    from pybert.test.predictor import Predictor
    data = TaskData()
    targets, sentences = data.read_data(raw_data_path=config['test_path'],
                                        aug_data_path=config['aug_test_path'],
                                        preprocessor=EnglishPreProcessor(),
                                        is_train=False,
                                        is_augament=False)
    lines = list(zip(sentences, targets))
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}

    test_data = processor.get_test(lines=lines)
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
    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))

    # ----------- predicting
    logger.info('model predicting....')
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu=args.n_gpu,
                          test_metrics=[
                              AUC(task_type='binary', average='macro',search_thresh=True),
                              Precision(task_type='binary', average='macro',search_thresh=True),  # 同样添加精确度指标
                              Recall(task_type='binary', average='macro',search_thresh=True),  # 同样添加召回率指标
                              HammingScore(search_thresh=True),  # 同样添加汉明分数指标
                              HammingLoss(search_thresh=True),  # 同样添加汉明损失指标
                              Jaccard(average='macro',search_thresh = True),  # 同样添加 jaccard 指标
                              F1Score(thresh=0.5, normalizate=True, task_type='binary', average='macro',
                                      search_thresh=True),  # 同样添加 F1 分数指标
                              MultiLabelReport(id2label=id2label, average='macro', logger=logger),
                              ClassReport()
                          ])
    result = predictor.predict(data=test_dataloader)
    print(result)


# %%
def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='GTA3', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--valid_size", default=0.3, type=float)
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
    log_file_path = Path(config['log_dir']) / f'{args.arch}-{time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())}.log'
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
                                            preprocessor=EnglishPreProcessor(),
                                            is_train=True,
                                            is_augament=False)
        data.train_val_split(X=sentences, y=targets, shuffle=True, stratify=False,
                             valid_size=args.valid_size, data_dir=config['data_dir'],
                             data_name=args.data_name)
    if args.do_train:
        run_train(args)

    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main()

import torch
from ..callback.progressbar import ProgressBar
from ..common.tools import model_device
from ..common.tools import summary
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from torch.nn.utils import clip_grad_norm_
from ..train.losses import BCEWithLogitsLoss, FocalLossWithSmoothing, FocalLoss, BCEWithLogLoss  # 修改导入路径
from torch.utils.data import RandomSampler, DataLoader
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cuda:1"

device = torch.device(dev)  # 设置设备

class Trainer(object):
    def __init__(self,args,model,logger,criterion,optimizer,scheduler,early_stopping,epoch_metrics,
                 batch_metrics,verbose = 1,training_monitor = None,model_checkpoint = None
                 ):
        self.args = args
        self.model = model
        self.logger =logger
        self.verbose = verbose
        # 确保使用我们自定义的BCEWithLogitsLoss
        self.criterion = BCEWithLogLoss()
        # self.criterion = BCEWithLogitsLoss(num_classes=4, gamma=0)
        # self.criterion = FocalLoss(gamma=2, alpha = 0.25)


        # 使用FocalLossWithSmoothing
        # self.criterion = FocalLossWithSmoothing(
        #     gamma=0.5,  # Focal Loss的聚焦参数
        #     epsilon=0.1,  # 标签平滑因子
        #     num_classes=4  # 标签数量
        # )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.start_epoch = 1
        self.global_step = 0
        self.model, self.device = model_device(n_gpu = args.n_gpu, model=self.model)
        
        # 设置标签权重
        if hasattr(self.criterion, 'set_label_weights') and hasattr(args, 'label_weights'):
            try:
                self.criterion.set_label_weights(args.label_weights)
                self.logger.info(f"Label weights set: {args.label_weights}")
            except Exception as e:
                self.logger.warning(f"Failed to set label weights: {str(e)}")
        
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if args.resume_path:
            self.logger.info(f"\nLoading checkpoint: {args.resume_path}")
            resume_dict = torch.load(args.resume_path / 'checkpoint_info.bin')
            best = resume_dict['best']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{args.resume_path}' and epoch {self.start_epoch} loaded")

    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()

    def save_info(self,epoch,best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model":model_save,
                 'epoch':epoch,
                 'best':best}
        return state

    def valid_epoch(self,data):
        pbar = ProgressBar(n_total=len(data),desc="Evaluating")
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids,input_mask).to(device)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            pbar(step=step)
        self.outputs = torch.cat(self.outputs, dim = 0).cpu().detach()
        self.targets = torch.cat(self.targets, dim = 0).cpu().detach()
        loss = self.criterion(target = self.targets, output=self.outputs)
        self.result['valid_loss'] = loss.item()
        print("------------- valid result --------------")
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'valid_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    # 加入标签平滑的二元交叉熵损失函数
    def label_smoothing_crossentropy(self, logits, target, epsilon=0.05, num_classes=4):
        """
        生成平滑标签
        Args:
            logits: 模型输出 shape: [batch_size, num_classes]
            target: 原始标签 shape: [batch_size, num_classes]
            epsilon: 平滑参数
            num_classes: 类别数量
        Returns:
            smooth_targets: 平滑后的标签 shape: [batch_size, num_classes]
        """
        smooth_value = 1 - epsilon
        epsilon_value = epsilon / (num_classes - 1)
        
        # 直接使用 torch 操作而不是循环，提高效率
        smooth_targets = torch.full_like(logits, epsilon_value)
        smooth_targets[target == 1] = smooth_value
        
        return smooth_targets

    def train_epoch(self, data):
        pbar = ProgressBar(n_total=len(data), desc='Training')
        tr_loss = AverageMeter()
        self.epoch_reset()
        
        for step, batch in enumerate(data):
            self.batch_reset()
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = self.model(input_ids, segment_ids, input_mask).to(device)
            smooth_targets = self.label_smoothing_crossentropy(logits, label_ids)
            # 计算损失，传入标签权重
            loss = self.criterion(output=logits, target=smooth_targets)
            
            if len(self.args.n_gpu) >= 2:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            if self.args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.args.grad_clip)
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            self.info['loss'] = loss.item()
            tr_loss.update(loss.item(), n=1)
            
            if self.verbose >= 1:
                pbar(step=step, info=self.info)
            
            # 累积输出和目标
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            
        print("\n------------- train result --------------")
        # 在epoch结束时拼接所有batch的结果
        self.outputs = torch.cat(self.outputs, dim=0).detach().cpu()
        self.targets = torch.cat(self.targets, dim=0).detach().cpu()
        
        self.result['loss'] = tr_loss.avg
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'{metric.name()}'] = value
                
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self,train_data,valid_data):
        self.model.zero_grad()
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 a
        for epoch in range(self.start_epoch,self.start_epoch+self.args.epochs):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            train_log = self.train_epoch(train_data)
            valid_log = self.valid_epoch(valid_data)
            '''
            使用 ** 操作符将valid_metrics解包传递给dict()时，
            valid_log中的键值对会覆盖了train_log中的同名项，有多的项会加入字典
            例如：
            train_metrics = {'accuracy': 0.90, 'precision': 0.85}
            valid_metrics = {'accuracy': 0.88, 'recall': 0.78}
            {'accuracy': 0.88, 'precision': 0.85, 'recall': 0.78}
            '''
            logs = dict(train_log,**valid_log)
            # 创建日志信息，根据value类型选择合适的格式化方式
            log_items = []
            for key, value in logs.items():
                if isinstance(value, dict):
                    # 如果是字典，不格式化，只显示键
                    log_items.append(f' {key}: dict ')
                elif value is None:
                    # 如果是None，显示N/A
                    log_items.append(f' {key}: N/A ')
                elif isinstance(value, (int, float)):
                    # 如果是数值，使用.4f格式化
                    log_items.append(f' {key}: {value:.4f} ')
                else:
                    # 其他类型直接转换为字符串
                    log_items.append(f' {key}: {str(value)} ')
            
            show_info = f'\nEpoch: {epoch} - ' + "-".join(log_items)
            self.logger.info(show_info)

            # save
            if self.training_monitor:
                self.training_monitor.epoch_step(logs) 

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch,best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.bert_epoch_step(current=logs[self.model_checkpoint.monitor],state = state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break

        # print("model summary info: ")
        # for step, (input_ids, input_mask, segment_ids, label_ids) in enumerate(train_data):
        #     input_ids = input_ids.to(self.device)
        #     input_mask = input_mask.to(self.device)
        #     segment_ids = segment_ids.to(self.device)
        #     summary(self.model,*(input_ids, segment_ids,input_mask),show_input=True)
        #     break
        # # ***************************************************************

def train(args, model, train_dataset, dev_dataset=None, test_dataset=None):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    # 计算每个标签的权重
    label_counts = train_dataset.get_label_counts()
    total_samples = len(train_dataset)
    label_weights = [count / total_samples for count in label_counts]
    
    # 设置损失函数
    criterion = BCEWithLogitsLoss()
    # 为每个标签设置不同的权重
    criterion.set_label_weights(label_weights)
    
    # ... rest of the training code ...





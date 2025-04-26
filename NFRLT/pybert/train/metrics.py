r"""Functional interface"""
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, hamming_loss

__call__ = ['Accuracy','AUC','F1Score','EntityScore','ClassReport','MultiLabelReport','AccuracyThresh', 'Precision', 'Recall', 'HammingScore', 'HammingLoss' ,'Jaccard']

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class Accuracy(Metric):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    Examples:
        >>> metric = Accuracy(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK
        self.reset()

    def __call__(self, logits, target):
        _, pred = logits.topk(self.topK, 1, True, True)
        pred = pred.t()
        # 计算预测值与目标值是否相等，得到一个布尔矩阵
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        self.correct_k = correct[:self.topK].view(-1).float().sum(0)
        self.total = target.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k)  / self.total

    def name(self):
        return 'accuracy'


class AccuracyThresh(Metric):
    '''
    计算给定阈值下的准确度
    可以使用topK参数设定计算K准确度
    Example:
        >>> metric = AccuracyThresh(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''
    def __init__(self,thresh = 0.5):
        super(AccuracyThresh,self).__init__()
        self.thresh = thresh
        self.reset()

    def __call__(self, logits, target):
        # 如果logits是列表，取最后一个元素（通常是最终的logits）
        if isinstance(logits, list):
            logits = logits[-1]
        self.y_pred = logits.sigmoid()
        self.y_true = target

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        data_size = self.y_pred.size(0)
        acc = np.mean(((self.y_pred>self.thresh)==self.y_true.byte()).float().cpu().numpy(), axis=1).sum()
        return acc / data_size

    def name(self):
        return 'accuracy'

class Precision(Metric):
    def __init__(self, task_type='binary', average='samples'):
        super(Precision, self).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.task_type = task_type
        self.average = average

    def __call__(self, logits, target):
        if self.task_type == 'binary':
            self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        elif self.task_type == 'multiclass':
            self.y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        precision = precision_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return precision

    def name(self):
        return 'precision'


class Recall(Metric):
    def __init__(self, task_type='binary', average='samples'):
        super(Recall, self).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.task_type = task_type
        self.average = average

    def __call__(self, logits, target):
        if self.task_type == 'binary':
            self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        elif self.task_type == 'multiclass':
            self.y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        recall = recall_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return recall

    def name(self):
        return 'recall'




class HammingScore(Metric):
    def __init__(self):
        super(HammingScore, self).__init__()

    def __call__(self, logits, target):
        self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        # 计算每个样本预测正确的标签比例的平均值
        correct_pred = (self.y_true == self.y_pred).astype(float)
        return np.mean(correct_pred)

    def name(self):
        return 'hamming_score'


class HammingLoss(Metric):
    def __init__(self):
        super(HammingLoss, self).__init__()

    def __call__(self, logits, target):
        self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        hamming_loss_val = hamming_loss(y_true=self.y_true, y_pred=self.y_pred)
        return hamming_loss_val

    def name(self):
        return 'hamming_loss'


class Jaccard(Metric):
    def __init__(self, average='macro'):  # 修改默认值为'macro'而不是'None'
        super(Jaccard, self).__init__()
        self.average = average

    def __call__(self, logits, target):
        self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        jaccard_val = jaccard_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return jaccard_val

    def name(self):
        return 'jaccard'  # 修正名称


class AUC(Metric):
    '''
    AUC score
    micro:
            全局计算指标，将标签指示矩阵的每个元素视为一个标签，计算全局的性能指标。
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            对每个标签分别计算指标，然后取它们的未加权平均值。这种方法不考虑标签不平衡的情况。
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            对每个标签分别计算指标，然后以标签的支持度（每个标签的真实实例数）加权平均。这种方法考虑了标签不平衡的情况。
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            对每个实例分别计算指标，然后取它们的平均值。
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = AUC(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self,task_type = 'binary',average = 'samples'):
        super(AUC, self).__init__()

        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.task_type = task_type
        self.average = average

    def __call__(self,logits,target):
        '''
        计算整个结果
        '''
        if self.task_type == 'binary':
            self.y_prob = (logits.sigmoid()>0.5).int().cpu().numpy()
        else:
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def value(self):
        '''
        计算指标得分
        '''
        auc = roc_auc_score(y_score=self.y_prob, y_true=self.y_true, average=self.average)
        return auc

    def name(self):
        return 'auc'

class F1Score(Metric):
    '''
    F1 Score
    binary:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = F1Score(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''
    def __init__(self, thresh=0.5, normalizate=True, task_type='binary', average='samples', search_thresh=False):
        super(F1Score, self).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self, y_prob):
        '''
        对于f1评分的指标，一般我们需要对阈值进行调整，一般不会使用默认的0.5值，因此
        这里我们对Thresh进行优化
        :return:
        '''
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            y_pred = (y_prob > threshold).astype(int)
            score = f1_score(y_true=self.y_true, y_pred=y_pred, average=self.average)
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def __call__(self, logits, target):
        '''
        计算整个结果
        :return:
        '''
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.search_thresh:
                best_thresh, _ = self.thresh_search(y_prob=y_prob)
                self.y_pred = (y_prob > best_thresh).astype(int)
            else:
                self.y_pred = (y_prob > self.thresh).astype(int)
        else:  # multiclass
            self.y_pred = np.argmax(y_prob, 1)

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        '''
        计算指标得分
        '''
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

    def name(self):
        return 'f1'

class ClassReport(Metric):
    '''
    classification report
    '''
    def __init__(self, logger=None):
        super(ClassReport, self).__init__()
        self.logger = logger

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def __call__(self, logits, target):
        # _, y_pred = torch.max(logits.data, 1)
        self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        self.y_true = target.cpu().numpy()

    def value(self):
        '''
        计算指标得分
        '''
        try:
            # 获取分类报告
            report_str = classification_report(
                y_true=self.y_true,
                y_pred=self.y_pred,
                target_names=['Usa', 'Sup', 'Dep', 'Per']
            )
            
            # 打印到控制台
            print(f"\n分类报告: \n{report_str}")
            
            # 如果有logger，也记录到日志
            if self.logger:
                self.logger.info(f"\n分类报告: \n{report_str}")
            
            # 同时获取字典形式的报告，用于程序处理
            report_dict = classification_report(
                y_true=self.y_true,
                y_pred=self.y_pred,
                target_names=['Usa', 'Sup', 'Dep', 'Per'],
                output_dict=True
            )
            
            return {
                'report': report_dict,
                'report_str': report_str
            }
        except Exception as e:
            error_msg = f"计算分类报告时出错: {str(e)}"
            print(error_msg)
            if self.logger:
                self.logger.error(error_msg)
            return None

    def name(self):
        return "class_report"

class MultiLabelReport(Metric):
    '''
    multi label report
    '''
    def __init__(self,id2label,average, logger):
        super(MultiLabelReport).__init__()
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted', 'None']
        self.id2label = id2label
        self.average = average
        self.logger = logger


    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def __call__(self,logits,target):
        self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        self.y_true = target.cpu().numpy()

    def value(self):
        '''
        计算指标得分
        '''
        # 计算每个标签的指标
        precisions = precision_score(self.y_true, self.y_pred, average=None)
        recalls = recall_score(self.y_true, self.y_pred, average=None)
        f1_scores = f1_score(self.y_true, self.y_pred, average=None)
        
        # 计算整体的指标
        hamming_loss_value = hamming_loss(self.y_true, self.y_pred)
        jaccard_value = jaccard_score(self.y_true, self.y_pred, average=self.average)
        
        # 计算hamming score (正确预测的标签比例)
        hamming_score = np.mean((self.y_true == self.y_pred).astype(float))

        # 打印每个标签的详细指标
        for i in range(len(precisions)):
            label_name = self.id2label.get(i, "Label")
            log_messages = '\n'.join([
                f"Label {label_name}:",
                f"  Precision={precisions[i]:.4f}",
                f"  Recall={recalls[i]:.4f}",
                f"  F1-Score={f1_scores[i]:.4f}"
            ])
            self.logger.info(log_messages)

        # 打印整体指标
        overall_metrics = '\n'.join([
            f"Overall Metrics:",
            f"  Hamming Loss: {hamming_loss_value:.4f}",
            f"  Hamming Score: {hamming_score:.4f}",
            f"  Jaccard Score: {jaccard_value:.4f}"
        ])
        self.logger.info(overall_metrics)



    def name(self):
        return "multilabel_report"


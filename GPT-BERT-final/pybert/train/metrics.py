r"""Functional interface"""
import torch
from sklearn.preprocessing import label_binarize, MultiLabelBinarizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold

__call__ = ['Accuracy','AUC','F1Score','EntityScore','ClassReport','MultiLabelReport','AccuracyThresh', 'Precision', 'Recall', 'HammingScore', 'HammingLoss','Jaccard']

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
    '''
    def __init__(self,search_thresh = True):
        super(AccuracyThresh,self).__init__()
        self.search_thresh = search_thresh
        self.reset()

    # def __call__(self, logits, target):
    #     self.y_pred = logits.sigmoid()
    #     self.y_true = target

    def __call__(self, logits, target):
        self.y_true = target
        y_prob = logits.sigmoid()
        if self.search_thresh == True:
            thresh, acc = self.thresh_search(y_prob=y_prob)
            return acc
        else:
            self.y_pred = logits.sigmoid()

    def thresh_search(self, y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in np.arange(0.1, 1, 0.01):
            self.y_pred = (y_prob > threshold).int()
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        data_size = self.y_pred.size(0)
        acc = np.mean((self.y_pred==self.y_true.byte()).float().cpu().numpy(), axis=1).sum()
        return acc / data_size

    def name(self):
        return 'accuracy'

class Precision(Metric):
    def __init__(self, task_type='binary', average='samples',search_thresh=True):
        super(Precision, self).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.task_type = task_type
        self.average = average
        self.search_thresh = search_thresh

    def __call__(self,logits,target):
        '''
        计算整个结果
        '''
        self.y_true = target.cpu().numpy()
        if self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
            if self.search_thresh == True:
                thresh, Precision, average_score = self.thresh_search(y_prob=y_prob)
                return average_score
            else:
                self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()

        elif self.task_type == 'multiclass':
            self.y_pred = logits.softmax(-1).data.cpu().detach().numpy()

    # def thresh_search(self, y_prob):
    #     best_threshold = 0
    #     best_score = 0
    #     for threshold in np.arange(0.1, 1, 0.01):
    #         self.y_pred = (y_prob > threshold).astype(int)
    #         score = self.value()
    #         if score > best_score:
    #             best_threshold = threshold
    #             best_score = score
    #     return best_threshold, best_score
    def thresh_search(self, y_prob):
        best_threshold = 0
        best_score = 0
        scores = []
        for threshold in np.arange(0.1, 1, 0.1):
            self.y_pred = (y_prob > threshold).astype(int)
            score = self.value()
            scores.append(score)
            if score > best_score:
                best_threshold = threshold
                best_score = score
        average_score = sum(scores) / len(scores)
        return best_threshold, best_score, average_score

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        precision = precision_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return precision

    def name(self):
        return 'precision'


class Recall(Metric):
    def __init__(self, task_type='binary', average='samples',search_thresh=True):
        super(Recall, self).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.task_type = task_type
        self.average = average
        self.search_thresh = search_thresh

    def __call__(self,logits,target):
        '''
        计算整个结果
        '''
        self.y_true = target.cpu().numpy()
        if self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
            if self.search_thresh == True:
                thresh, recall, average_score = self.thresh_search(y_prob=y_prob)
                return average_score
            else:
                self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()

        elif self.task_type == 'multiclass':
            self.y_pred = logits.softmax(-1).data.cpu().detach().numpy()

    def thresh_search(self, y_prob):
        best_threshold = 0
        best_score = 0
        scores = []
        for threshold in np.arange(0.1, 1, 0.01):
            self.y_pred = (y_prob > threshold).astype(int)
            score = self.value()
            scores.append(score)
            if score > best_score:
                best_threshold = threshold
                best_score = score
        average_score = sum(scores) / len(scores)
        return best_threshold, best_score, average_score

    # def thresh_search(self, y_prob):
    #     best_threshold = 0
    #     best_score = 0
    #     for threshold in np.arange(0.1, 1, 0.01):
    #         self.y_pred = (y_prob > threshold).astype(int)
    #         score = self.value()
    #         if score > best_score:
    #             best_threshold = threshold
    #             best_score = score
    #     return best_threshold, best_score

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        recall = recall_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return recall

    def name(self):
        return 'recall'


def hamming_score1(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 0
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

class HammingScore(Metric):
    def __init__(self,search_thresh=True):
        super(HammingScore, self).__init__()
        self.search_thresh = search_thresh


    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        y_prob = logits.sigmoid().data.cpu().numpy()
        if self.search_thresh == True:
            thresh, hammingscore = self.thresh_search(y_prob=y_prob)
            return hammingscore
        else:
            self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()

    def thresh_search(self, y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in np.arange(0.01, 1, 0.01):
            self.y_pred = (y_prob > threshold).astype(int)
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def reset(self):
        self.y_pred = 0
        self.y_true = 0


    def value(self):
        hamming_score = hamming_score1(y_true=self.y_true, y_pred=self.y_pred)
        return hamming_score

    def name(self):
        return 'hamming_score'


class HammingLoss(Metric):
    def __init__(self,search_thresh=True):
        super(HammingLoss, self).__init__()
        self.search_thresh = search_thresh

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        y_prob = logits.sigmoid().data.cpu().numpy()
        if self.search_thresh == True:
            thresh, hammingloss, average_score = self.thresh_search(y_prob=y_prob)
            return average_score
        else:
            self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()

    def thresh_search(self, y_prob):
        best_threshold = 0
        best_score = 0
        scores = []
        for threshold in np.arange(0.01, 1, 0.01):
            self.y_pred = (y_prob > threshold).astype(int)
            score = self.value()
            scores.append(score)
            if score < best_score:
                best_threshold = threshold
                best_score = score
        average_score = sum(scores)/len(scores)
        return best_threshold, best_score, average_score

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        hamming_loss_val = hamming_loss(y_true=self.y_true, y_pred=self.y_pred)
        return hamming_loss_val

    def name(self):
        return 'hamming_loss'


class Jaccard(Metric):
    def __init__(self,average='None',search_thresh = True):
        super(Jaccard, self).__init__()
        self.average = average
        self.search_thresh = search_thresh

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        y_prob = logits.sigmoid().data.cpu().numpy()
        if self.search_thresh == True:
            thresh, jac = self.thresh_search(y_prob=y_prob)
            return jac
        else:
            self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()

    def thresh_search(self, y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in np.arange(0.01, 1, 0.01):
            self.y_pred = (y_prob > threshold).astype(int)
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def reset(self):
        self.y_pred = 0
        self.y_true = 0


    def value(self):
        jaccard_val = jaccard_score(y_true=self.y_true, y_pred=self.y_pred,average=self.average)
        return jaccard_val

    def name(self):
        return 'jac'

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
    '''

    def __init__(self,task_type = 'binary',average = 'samples',search_thresh = True):
        super(AUC, self).__init__()

        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.task_type = task_type
        self.average = average
        self.search_thresh = search_thresh

    def __call__(self,logits,target):
        '''
        计算整个结果
        '''
        self.y_true = target.cpu().numpy()
        if self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
            if self.search_thresh == True:
                thresh, auc = self.thresh_search(y_prob=y_prob)
                return auc
            else:
                self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()

        elif self.task_type == 'multiclass':
            self.y_pred = logits.softmax(-1).data.cpu().detach().numpy()

    def thresh_search(self, y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in np.arange(0.01, 1, 0.01):
            self.y_pred = (y_prob > threshold).astype(int)
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        '''
        计算指标得分
        '''
        auc = roc_auc_score(y_score=self.y_pred, y_true=self.y_true, average=self.average)
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
    '''
    def __init__(self,thresh = 0.5, normalizate = True,task_type = 'binary',average = 'None',search_thresh = False):
        super(F1Score).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self,y_prob):
        '''
        对于f1评分的指标，一般我们需要对阈值进行调整，一般不会使用默认的0.5值，因此
        这里我们对Thresh进行优化
        :return:
        '''
        best_threshold = 0
        best_score = 0
        for threshold in np.arange(0.01, 1, 0.01):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
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
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh ).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                return f1

        if self.task_type == 'multiclass':
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
    class report
    '''
    def __init__(self):
        super(ClassReport).__init__()
        self.target_names = ['Usa', 'Sup', 'Dep', 'Per']


    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def __call__(self,logits,target):

        # 转换成了一维的元组，错了
        # self.y_pred = torch.max(logits, 1).indices.cpu().numpy()
        self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        self.y_true = target.cpu().numpy()




    def value(self):
        '''
        计算指标得分
        '''
        score = classification_report(y_true = self.y_true,
                                      y_pred = self.y_pred,
                                      target_names=self.target_names)



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

    def reset_thresh(self):
        self.thresh = 0
        return self.thresh

    def reset_score(self):
        self.score = 0
        return self.score

    def __call__(self,logits,target):
        self.y_prob = logits.sigmoid().data.cpu().numpy()
        # self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        self.y_true = target.cpu().numpy()

    def value(self):
        '''
        计算指标得分
        '''

        #分别计算每个标签的每个评价指标的阈值及最佳分数
        for i, label in self.id2label.items():
            best_threshold = []
            best_score = []
            bestscore = 0
            thresh = 0
            for threshold in np.arange(0.1, 1, 0.1):
                self.y_pred = (self.y_prob > threshold).astype(int)
                auc = roc_auc_score(y_true=self.y_true[:, i],y_score=self.y_pred[:, i])
                if auc > bestscore:
                    thresh = threshold
                    bestscore = auc
            best_threshold.append(thresh)
            best_score.append(bestscore)

            # hamming损失使用均值计算，而不用最佳阈值，因为阈值为0时，损失也为0
            thresh = self.reset_thresh()
            scores = []
            for threshold in np.arange(0, 1, 0.1):
                self.y_pred = (self.y_prob > threshold).astype(int)
                hamming_loss_val = hamming_loss(y_true=self.y_true[:, i], y_pred=self.y_pred[:, i])
                scores.append(hamming_loss_val)
            average_score = sum(scores)/len(scores)
            best_threshold.append(thresh)
            best_score.append(average_score)

            bestscore = self.reset_score()
            thresh = self.reset_thresh()
            for threshold in np.arange(0.1, 1, 0.1):
                self.y_pred = (self.y_prob > threshold).astype(int)
                hamming_score = hamming_score1(y_true=self.y_true[:, i], y_pred=self.y_pred[:, i])
                if hamming_score > bestscore:
                    thresh = threshold
                    bestscore = hamming_score
            best_threshold.append(thresh)
            best_score.append(bestscore)

            bestscore = self.reset_score()
            thresh = self.reset_thresh()
            for threshold in np.arange(0.1, 1, 0.1):
                self.y_pred = (self.y_prob > threshold).astype(int)
                precision = precision_score(y_true=self.y_true[:, i], y_pred=self.y_pred[:, i], average=self.average)
                if precision > bestscore:
                    thresh = threshold
                    bestscore = precision
            best_threshold.append(thresh)
            best_score.append(bestscore)

            bestscore = self.reset_score()
            thresh = self.reset_thresh()
            for threshold in np.arange(0.1, 1, 0.1):
                self.y_pred = (self.y_prob > threshold).astype(int)
                recall = recall_score(y_true=self.y_true[:, i], y_pred=self.y_pred[:, i], average=self.average)
                if recall > bestscore:
                    thresh = threshold
                    bestscore = recall
            best_threshold.append(thresh)
            best_score.append(bestscore)

            bestscore = self.reset_score()
            thresh = self.reset_thresh()
            for threshold in np.arange(0.1, 1, 0.1):
                self.y_pred = (self.y_prob > threshold).astype(int)
                f1 = f1_score(y_true=self.y_true[:, i], y_pred=self.y_pred[:, i], average=self.average, zero_division=0)
                if f1 > bestscore:
                    thresh = threshold
                    bestscore = f1
            best_threshold.append(thresh)
            best_score.append(bestscore)

            log_messages = '\n'.join([f"Label: {label} - AUC: {best_score[0]:.4f}, Hamming Loss: {best_score[1]:.4f}, Hamming Score: {best_score[2]:.4f}, Precision: {best_score[3]:.4f}, Recall: {best_score[4]:.4f}, F1 Score: {best_score[5]:.4f}"])
            self.logger.info(log_messages)


    def name(self):
        return "multilabel_report"


#encoding:utf-8
import torch
import numpy as np
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar

class Predictor(object):
    def __init__(self,model,logger,n_gpu,test_metrics):
        self.model = model
        self.logger = logger
        self.test_metrics = test_metrics
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)

    def test_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.test_metrics:
            metric.reset()

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data),desc='Testing')
        self.test_reset()
        # all_logits = None
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask)
                # logits = logits.sigmoid()
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            pbar(step=step)
            # if all_logits is None:
            #     all_logits = logits.detach().cpu().numpy()
            # else:
            #     # 每个批次的结果按行累加
            #     all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
        print("\n------------- test result --------------")
        self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
        self.targets = torch.cat(self.targets, dim=0).cpu().detach()
        if self.test_metrics:
            for metric in self.test_metrics:
                if metric.name() == 'class_report' or metric.name() == 'multilabel_report':
                    metric(logits=self.outputs, target=self.targets)
                    value = metric.value()
                    if value:
                        self.result[f'test_{metric.name()}'] = value
                else:
                    value = metric(logits=self.outputs, target=self.targets)
                    if value:
                        self.result[f'test_{metric.name()}'] = value
        logs = dict(self.result)
        show_info = f'\nTest:' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        self.logger.info(show_info)

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return show_info









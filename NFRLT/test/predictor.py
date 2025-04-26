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
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            pbar(step=step)
        
        print("\n------------- test result --------------")
        self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
        self.targets = torch.cat(self.targets, dim=0).cpu().detach()
        
        # 计算并保存所有指标
        if self.test_metrics:
            for metric in self.test_metrics:
                # 对于所有指标，首先计算预测值和目标值
                metric(logits=self.outputs, target=self.targets)
                
                # 然后获取结果
                value = metric.value()
                
                # 在结果字典中保存值（如果值存在）
                if value is not None:
                    # 如果指标名称是 'class_report' 或 'multilabel_report'，可能有特殊处理
                    self.result[f'test_{metric.name()}'] = value
        
        # 打印结果摘要
        logs = {}
        for key, value in self.result.items():
            if isinstance(value, (int, float)):
                logs[key] = value
                
        if logs:
            show_info = f'\nTest:' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            self.logger.info(show_info)
        
        # 对于详细的报告，单独打印
        if 'test_multilabel_report' in self.result:
            report = self.result['test_multilabel_report']
            if isinstance(report, dict) and 'overall' in report:
                self.logger.info("\n整体性能:")
                for metric, value in report['overall'].items():
                    self.logger.info(f"  {metric}: {value:.4f}")
        
        # if 'test_class_report' in self.result and isinstance(self.result['test_class_report'], dict):
        #     if 'report_str' in self.result['test_class_report']:
        #         self.logger.info(f"\n分类报告:\n{self.result['test_class_report']['report_str']}")
        
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
            
        return self.result









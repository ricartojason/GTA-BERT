import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from .text_cnn import TextCNN

class BertForMultiLable(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLable, self).__init__(config)
        self.bert = BertModel(config)
        #dropout丢弃比率0.1，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # TextCNN 参数
        num_filters = 256  # 每个卷积核的通道数
        filter_sizes = [2, 3, 4]  # 卷积核大小
        
        # 使用 TextCNN 替代原来的线性层
        self.text_cnn = TextCNN(
            hidden_size=config.hidden_size,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            num_labels=config.num_labels,
            dropout=config.hidden_dropout_prob
        )
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        # 获取 BERT 的输出
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        # 使用序列输出 [batch_size, seq_len, hidden_size]
        sequence_output = outputs[0]  # 使用第一个输出，即序列输出
        
        # 应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        # 通过 TextCNN 进行分类
        logits = self.text_cnn(sequence_output)
        
        return logits

    def unfreeze(self,start_layer,end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertPreTrainedModel, BertModel

class BertForMultiLable(BertPreTrainedModel):
    # def __init__(self, config):
    #     super(BertForMultiLable, self).__init__(config)
    #     self.bert = BertModel(config)
    #     #dropout丢弃比率0.1，防止过拟合
    #     self.dropout = nn.Dropout(config.hidden_dropout_prob)
    #     self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    #     self.init_weights()
    #
    # def forward(self, input_ids, token_type_ids=None, attention_mask=None,head_mask=None):
    #     outputs = self.bert(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, head_mask=head_mask)
    #     pooled_output = outputs[1]
    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)
    #     return logits

    def __init__(self, config):
        super(BertForMultiLable, self).__init__(config)
        self.bert = BertModel(config)
        #BERT嵌入层后跟三个不同距离滑动窗口的TextCNN
        self.conv1 = nn.Conv2d(1, config.max_length, (2, config.hidden_size))
        self.conv2 = nn.Conv2d(1, config.max_length, (3, config.hidden_size))
        self.conv3 = nn.Conv2d(1, config.max_length, (4, config.hidden_size))
        self.linear = nn.Linear(config.hidden_size * 3, config.num_labels)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # pooled_output = outputs[1]
        sequence_output = outputs[0]
        print(sequence_output.shape)
        # 为卷积操作准备输入，增加一个维度以模拟通道
        out = sequence_output.unsqueeze(1)  # 形状变为 (batch_size, 1, sequence_length, hidden_size)
        out1 = self.conv_and_pool(self.conv1, out)
        out2 = self.conv_and_pool(self.conv2, out)
        out3 = self.conv_and_pool(self.conv3, out)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.linear(out)

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
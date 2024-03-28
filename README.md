# GTA-BERT
一种使用改进的BERT模型进行非功能需求多标签分类的方法（A NFRs Multi Label Classification Method Using Enhanced Bert)

## 运行说明：
运行项目首先需要在[Bert-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main)中下载pytorch_model.bin, config.json, vocab.txt.
- `bert-base-uncased-pytorch_model.bin` 修改名字为 `pytorch_model.bin`
- `bert-base-uncased-config.json` 修改名字为 `config.json`
- `bert-base-uncased-vocab.txt` 修改名字为 `bert_vocab.txt`
并将上述文件存放到pretrain\bert\base-uncansed中。

/configs/basic_config用于修改模型参数、数据集存放的路径

/output/log用于存放训练模型过程中的模型参数、训练|测试结果
/output/figure用于存放训练、验证过程中评价指标的变化

运行`python run_bert.py --do_data`以处理数据
运行`python run_bert.py --do_train --save_best --do_lower_case`运行、微调模型
运行`run_bert.py --do_test --do_lower_case`在测试集上评估效果

## 数据集说明：
数据集保存路径为/dataset

review_origin_train.scv是原始训练集

review_origin_test.scv是原始测试集

带有TTA前缀的数据集都是GPT for Test-Time augmentation（TTA）生成的样本

运行TTA需要在run_bert.py中将data.read_data()的is_augament参数设置为True

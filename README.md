# NRMLC-BT
利用基于迁移学习的 BERT-TextCNN 对 NFR 进行多标签分类（Multi-label Classification of NFR
with Transfer Learning-Based BERT-TextCNN)

## 运行说明(Running Instructions)：
1.To run the project, you first need to download pytorch_model.bin, config.json, vocab.txt.json. in [Bert - base] (https://huggingface.co/google-bert/bert-base-uncased/tree/main).

2.Place `model` ,`config` and `vocab` file into  the `/pybert/pretrain/bert/base-uncased` directory.

3.Download dataset in this github(https://github.com/ricartojason/GTA-BERT/tree/main/Dataset) and place in `pybert/dataset`.

4./configs/basic_config is used to change the model parameters and the path where the dataset is stored

5./output/log is used to store model parameters and train/test results during model training

6./output/figure is used to store the changes of evaluation indexes during training and validation

7.Run 'python run bert.py --do data' to process the data

8.Run 'python run_bert.py --do_train --save_best --do_lower_case' to train, validate, and fine-tune the model

9.Run 'run_bert.py --do_test --do_lower_case' to evaluate the performance on the test set

10.To run TTA, set the is_augament parameter of the data.read_data() function to True in run_bert.py.

## 数据集说明(Description of the data set)：
1.Our MNRDataset has a total of 11,700 app reviews

2.review origin train.scv is the original training set of Jha and Mahmoud, which we refer to in this paper as MNR-1. 

3.review origin Test.scv is the original test set of Jha and Mahmoud, which we refer to in this paper as MNR-2 and MNR-3. 

4.In the Labelled datasets folder, we annotated 4600 reviews of 8 apps in the Quim Motger dataset, which we refer to in this paper as MNR-4.

5.We re-annotated MNR-1 dataset as a single-label dataset called review_single_table(6000 reviews) and compared with to verify the validity of Multi Label Dataset for NFRs classification.

6.MNR_Data is the training set after integrating the new dataset(10100 reviews), while MNR_Test is the test set after integrating the new dataset(1600 reviews).

7.The datasets prefixed with TTA are all samples generated by GPT for Train/Test-Time augmentation (TTA).

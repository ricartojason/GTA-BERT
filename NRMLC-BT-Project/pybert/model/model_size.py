#
# BertForMultiLable(
#   (bert): BertModel(
#     (embeddings): BertEmbeddings(
#       (word_embeddings): Embedding(30522, 768, padding_idx=0)
#       (position_embeddings): Embedding(512, 768)
#       (token_type_embeddings): Embedding(2, 768)
#       (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#       (dropout): Dropout(p=0.1, inplace=False)
#     )
#     (encoder): BertEncoder(
#       (layer): ModuleList(
#         (0-11): 12 x BertLayer(
#           (attention): BertAttention(
#             (self): BertSelfAttention(
#               (query): Linear(in_features=768, out_features=768, bias=True)
#               (key): Linear(in_features=768, out_features=768, bias=True)
#               (value): Linear(in_features=768, out_features=768, bias=True)
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#             (output): BertSelfOutput(
#               (dense): Linear(in_features=768, out_features=768, bias=True)
#               (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#           )
#           (intermediate): BertIntermediate(
#             (dense): Linear(in_features=768, out_features=3072, bias=True)
#           )
#           (output): BertOutput(
#             (dense): Linear(in_features=3072, out_features=768, bias=True)
#             (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#         )
#       )
#     )
#     (pooler): BertPooler(
#       (dense): Linear(in_features=768, out_features=768, bias=True)
#       (activation): Tanh()
#     )
#   )
#   (dropout): Dropout(p=0.1, inplace=False)
#   (classifier): Linear(in_features=768, out_features=4, bias=True)
# )

# 使用概率阈值
# 而不是使用固定的0
# .5
# 阈值，可以为每个标签设置不同的阈值。
# 通过交叉验证来找到每个标签的最佳阈值。
# 2.
# 动态阈值调整
# 根据验证集上的性能指标动态调整阈值。
# 使用如F1分数这样的平衡指标来确定最佳阈值。
# 3.
# ROC曲线和AUC
# 利用ROC曲线（接收者操作特征曲线）和AUC（曲线下面积）来评估不同阈值的性能。
# 选择在ROC曲线上提供最佳分类性能的阈值。
# 4.
# PR曲线和AP
# 对于多标签分类，PR曲线（精确率 - 召回率曲线）和AP（平均精确率）是更合适的评估工具。
# 选择最大化AP的阈值。
# 5.
# 阈值搜索
# 进行网格搜索或随机搜索来找到最佳阈值。
# 可以结合交叉验证来进行更系统的搜索。
# 6.
# 标签权重
# 如果某些标签比其他标签更重要，可以为它们设置更高的阈值。
# 可以使用基于标签重要性的自适应阈值方法。
# 7.
# 阈值优化算法
# 使用优化算法（如梯度下降、遗传算法等）来找到最佳阈值。
# 8.
# 集成多个阈值
# 对于每个标签使用不同的阈值，甚至可以为同一标签在不同情况下使用不同的阈值。
# 实际操作示例
# 以下是一个简单的示例，展示如何使用交叉验证来为每个标签找到最佳阈值：
#
# from sklearn.metrics import f1_score
# from sklearn.model_selection import KFold
#
# # 假设 logits 是模型输出的原始概率分数，target 是真实标签
# # KFold 交叉验证
# kf = KFold(n_splits=5)
#
# best_thresholds = {}
# for label in range(num_labels):
#     thresholds = np.linspace(0, 1, num=100)  # 定义一个阈值范围
#     f1_scores = []
#     for train_index, val_index in kf.split(logits):
#         # 划分训练集和验证集
#         X_train, X_val = logits[train_index], logits[val_index]
#         y_train, y_val = target[train_index], target[val_index]
#
#         # 遍历所有阈值
#         for threshold in thresholds:
#             y_pred = (X_val > threshold).astype(int)
#             f1 = f1_score(y_val, y_pred, average='binary', labels=[label])
#             f1_scores.append(f1)
#
#     # 选择验证集上F1分数最高的阈值
#     best_thresholds[label] = thresholds[np.argmax(f1_scores)]
#
# # 使用找到的最佳阈值进行预测
# for label in range(num_labels):
#     threshold = best_thresholds[label]
#     predictions = (logits > threshold).astype(int)
#     # 计算测试集上的最终指标

# import numpy as np
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score
# # 假设X是你的特征矩阵，Y是你的多标签标签矩阵
# # X = ... # Y = ...
# # 初始化随机森林分类器
# classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# # 初始化k折交叉验证，这里使用5折
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # 用于存储每个阈值的平均F1分数
# f1_scores = []
# # 遍历所有阈值
# for threshold in np.arange(0.1, 1.0, 0.1):
#     # 初始化F1分数列表
#     f1_scores_for_threshold = []
#     # 执行k折交叉验证
#     for train_index, test_index in kfold.split(data):
#         X_train, X_test = train_index[train_index], train_index[test_index]
#         y_train, y_test = test_index[train_index], test_index[test_index]
#         # 训练模型
#         classifier.fit(X_train, y_train)
#         # 预测概率
#         y_pred_proba = classifier.predict_proba(X_test)[:, 1]
#         # 假设我们只关心第一个类别的预测概率
#         # 应用阈值进行分类
#         y_pred = (y_pred_proba >= threshold).astype(int)
#         # 计算F1分数
#         f1 = f1_score(y_test, y_pred, average='micro')
#         # 或者使用其他平均方法
#         f1_scores_for_threshold.append(f1)
#         # 计算当前阈值的平均F1分数
#         average_f1 = np.mean(f1_scores_for_threshold)
#         f1_scores.append((threshold, average_f1))
#         # 找到最佳阈值
#         best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
#         print(f"最佳阈值: {best_threshold}, 平均F1分数: {best_f1}")

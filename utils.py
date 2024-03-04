import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator

pd.options.mode.chained_assignment = None


# 使用SMOTE算法平衡数据
def balance_data_with_smote(data, drop_columns, target_column, sampling_strategy='not majority', random_state=42):
    """
    使用SMOTE算法处理数据不平衡问题。
    :param random_state: 随机种子
    :param sampling_strategy: 采样方式
    :param data: 数据集
    :param drop_columns: 需要从数据集中删除的列
    :param target_column: 目标列名
    :return: 平衡后的特征集和目标列
    """
    X = data.drop([drop_columns, target_column], axis=1)
    y = data[target_column]
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced


# 生成K折伪标签
def generate_pseudo_labels(X_unlabeled, X_train, y_train, classifier: BaseEstimator, n_splits=5, random_state=42):
    """
    使用K折交叉验证生成伪标签。
    :param X_unlabeled: 未标记的数据集
    :param X_train: 训练数据集
    :param y_train: 训练数据集的目标值
    :param classifier: 用户指定的分类器实例
    :param n_splits: K折分割的数量
    :param random_state: 随机种子
    :return: 伪标签数组
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pseudo_labels = []
    for train_index, test_index in kf.split(X_unlabeled):
        X_train_fold, X_test_fold = X_unlabeled.iloc[train_index], X_unlabeled.iloc[test_index]
        X_combined = pd.concat([X_train, X_train_fold])
        y_combined = pd.concat([y_train, pd.Series([-1] * len(X_train_fold))])
        # 使用用户提供的分类器实例进行训练
        classifier.fit(X_combined, y_combined)
        pseudo_label = classifier.predict(X_test_fold)
        pseudo_labels.append(pseudo_label)
    return np.concatenate(pseudo_labels)


# 根据阈值删除难以分辨的训练样本
def filter_difficult_samples(X_train, y_train, model, thresholds):
    """
    根据给定阈值删除难以分辨的训练样本。
    :param X_train: 训练特征数据
    :param y_train: 训练标签数据
    :param model: 训练好的模型
    :param thresholds: 不同类别的难度阈值
    :return: 清洗后的训练特征数据和标签数据
    """
    # 对训练集进行预测，获取概率
    y_pred = model.predict_proba(X_train)

    # 复制预测结果用于计算第二大概率
    y_pred_copy = y_pred.copy()
    y_pred_copy[np.arange(y_pred_copy.shape[0]), np.argmax(y_pred_copy, axis=1)] = 0

    # 计算每个样本最大概率与第二大概率的差异
    diff = np.max(y_pred, axis=1) - np.max(y_pred_copy, axis=1)

    # 根据阈值找出难以分辨的样本索引
    difficult_samples_indices = {}
    for label, threshold in thresholds.items():
        difficult_samples_indices[label] = np.where((y_train == label) & (diff < threshold))[0]

    # 获取难以分辨的样本
    difficult_samples_X = {}
    difficult_samples_y = {}
    for label in thresholds.keys():
        difficult_samples_X[label], difficult_samples_y[label] = X_train.iloc[difficult_samples_indices[label]], \
            y_train.iloc[difficult_samples_indices[label]]

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    indices_to_drop = np.concatenate(list(difficult_samples_indices.values()))
    indices_to_drop = [index for index in indices_to_drop if index in X_train.index]

    X_train_cleaned = X_train.drop(indices_to_drop)
    y_train_cleaned = y_train.drop(indices_to_drop)

    return X_train_cleaned, y_train_cleaned


# 使用预测结果增强训练集并更新模型
def enhance_training_set_and_update_model(X_train, y_train, X_test, classifier: BaseEstimator, threshold=0.65):
    """
    使用模型预测结果来增强训练集，并更新模型。
    :param X_train: 原始训练数据集
    :param y_train: 原始训练数据集的目标值
    :param X_test: 测试数据集
    :param classifier: 用户指定的分类器实例
    :param threshold: 用于确定预测结果是否被接受加入训练集的阈值
    :return: 更新后的模型，以及用于预测的测试集
    """
    # 使用用户提供的分类器实例对测试集进行预测
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred_final = [np.argmax(pred) if max(pred) > threshold else -1 for pred in y_pred_proba]

    # 将预测结果合并到测试集中
    X_test['label'] = y_pred_final

    # 分离标记过的测试数据和未标记的测试数据
    X_test_labeled = X_test[X_test['label'] != -1]
    X_test_unlabeled = X_test[X_test['label'] == -1].drop('label', axis=1)

    # 将确定的预测结果加入训练集
    X_train_updated = pd.concat([X_train, X_test_labeled.drop('label', axis=1)])
    y_train_updated = pd.concat([y_train, X_test_labeled['label']])

    # 使用更新后的训练集重新训练模型
    classifier.fit(X_train_updated, y_train_updated)

    return classifier, X_test_labeled, X_test_unlabeled

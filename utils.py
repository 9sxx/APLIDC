import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator

pd.options.mode.chained_assignment = None


# 使用SMOTE算法平衡数据
def balance_data_with_smote(data, drop_columns, target_column, sampling_strategy='not majority', random_state=42, method='regular'):
    """
    使用SMOTE算法及其变种处理数据不平衡问题。
    :param data: 数据集
    :param drop_columns: 需要从数据集中删除的列
    :param target_column: 目标列名
    :param sampling_strategy: 采样方式
    :param random_state: 随机种子
    :param method: 使用的SMOTE算法变种，可选'regular', 'borderline', 'svm', 'adasyn'
    :return: 平衡后的特征集和目标列
    """
    X = data.drop([drop_columns, target_column], axis=1)
    y = data[target_column]

    if method == 'regular':
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == 'borderline':
        smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == 'svm':
        smote = SVMSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == 'adasyn':
        smote = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        raise ValueError("Unsupported SMOTE method")

    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced


# 生成K折伪标签
def generate_pseudo_labels(X_unlabeled, X_train, y_train, classifier: BaseEstimator, pseudo_label=-1, n_splits=5, random_state=42):
    """
    使用K折交叉验证生成伪标签，并将这些伪标签与未标记数据对应起来，最后返回一个包含数据和伪标签的DataFrame。
    :param pseudo_label: 伪标签
    :param X_unlabeled: 未标记的数据集
    :param X_train: 训练数据集
    :param y_train: 训练数据集的目标值
    :param classifier: 用户指定的分类器实例
    :param n_splits: K折分割的数量
    :param random_state: 随机种子
    :return: 包含伪标签的DataFrame
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # 初始化一个空的DataFrame来存储伪标签和数据
    pseudo_labeled_data = pd.DataFrame()
    for train_index, test_index in kf.split(X_unlabeled):
        X_train_fold, X_test_fold = X_unlabeled.iloc[train_index], X_unlabeled.iloc[test_index]
        X_combined = pd.concat([X_train, X_train_fold])
        y_combined = pd.concat([y_train, pd.Series([pseudo_label] * len(X_train_fold))])
        # 使用用户提供的分类器实例进行训练
        classifier.fit(X_combined, y_combined)
        # 生成伪标签
        pseudo_labels = classifier.predict(X_test_fold)

        # 只保留那些伪标签不为-1的样本
        filtered_indices = pseudo_labels != pseudo_label
        filtered_X_test_fold = X_test_fold[filtered_indices]
        filtered_pseudo_labels = pseudo_labels[filtered_indices]

        # 创建一个临时DataFrame来存储伪标签和对应的数据，并将其加入最终的DataFrame中
        temp_df = filtered_X_test_fold.copy()
        temp_df['pseudo_label'] = filtered_pseudo_labels
        pseudo_labeled_data = pd.concat([pseudo_labeled_data, temp_df])
    # 重新索引
    pseudo_labeled_data.reset_index(drop=True, inplace=True)
    return pseudo_labeled_data


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
def enhance_training_set_and_update_model(X_train, y_train, X_test, classifier: BaseEstimator, pseudo_labels=-1, threshold=0.65):
    """
    使用模型预测结果来增强训练集，并更新模型。
    :param pseudo_labels: 伪标签
    :param X_train: 原始训练数据集
    :param y_train: 原始训练数据集的目标值
    :param X_test: 测试数据集
    :param classifier: 用户指定的分类器实例
    :param threshold: 用于确定预测结果是否被接受加入训练集的阈值
    :return: 更新后的模型，增广后的训练集，增广后的训练集标签，确定标签的测试集以及待测的测试集
    """
    # 使用用户提供的分类器实例对测试集进行预测
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred_final = [np.argmax(pred) if max(pred) > threshold else pseudo_labels for pred in y_pred_proba]

    # 将预测结果合并到测试集中
    X_test['label'] = y_pred_final

    # 分离标记过的测试数据和未标记的测试数据
    X_test_labeled = X_test[X_test['label'] != pseudo_labels]
    X_test_unlabeled = X_test[X_test['label'] == pseudo_labels].drop('label', axis=1)

    # 将确定的预测结果加入训练集
    X_train_updated = pd.concat([X_train, X_test_labeled.drop('label', axis=1)])
    y_train_updated = pd.concat([y_train, X_test_labeled['label']])

    # 使用更新后的训练集重新训练模型
    classifier.fit(X_train_updated, y_train_updated)

    return classifier, X_train_updated, y_train_updated, X_test_labeled, X_test_unlabeled


import copy
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin


class APLIDCClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier: BaseEstimator,
                 use_smote=True, use_pseudo_labeling=True, use_pseudo_labeling_fit=False,
                 use_difficulty_filtering=True, use_enhance_training_set_and_update_model=True,
                 method='regular', pseudo_label=-1, pseudo_label_threshold=0.8,
                 difficulty_thresholds={}, random_state=42):
        self.base_classifier_template = base_classifier  # 保存原始分类器作为模板
        self.base_classifier = copy.deepcopy(base_classifier)  # 创建用于训练的分类器实例
        self.use_smote = use_smote
        self.use_pseudo_labeling = use_pseudo_labeling
        self.use_pseudo_labeling_fit = use_pseudo_labeling_fit
        self.use_difficulty_filtering = use_difficulty_filtering
        self.use_enhance_training_set_and_update_model = use_enhance_training_set_and_update_model
        self.method = method
        self.pseudo_label = pseudo_label
        self.pseudo_label_threshold = pseudo_label_threshold
        self.difficulty_thresholds = difficulty_thresholds
        self.random_state = random_state

    def fit(self, X_train, y_train, X_unlabeled):
        if self.use_smote:
            # Step 1: Balance data with SMOTE
            X_balanced, y_balanced = self._balance_data_with_smote(X_train, y_train)
        else:
            X_balanced, y_balanced = X_train, y_train

        if self.use_pseudo_labeling:
            # Step 2: Generate pseudo labels for unlabeled data
            pseudo_labeled_data = self._generate_pseudo_labels(X_unlabeled, X_balanced, y_balanced, self.pseudo_label)

            # 从pseudo_labeled_data中提取特征和伪标签，准备与原始训练集合并
            X_pseudo = pseudo_labeled_data.drop(['pseudo_label'], axis=1)
            y_pseudo = pseudo_labeled_data['pseudo_label']

            # 数据增广：将生成的带有伪标签的数据与原始平衡后的训练集合并
            X_augmented = pd.concat([X_balanced, X_pseudo])
            y_augmented = pd.concat([y_balanced, y_pseudo])
            if self.use_pseudo_labeling_fit:
                self.base_classifier = copy.deepcopy(self.base_classifier_template)  # 每次fit前重置分类器
                self.base_classifier.fit(X_augmented, y_augmented)
        else:
            self.base_classifier = copy.deepcopy(self.base_classifier_template)  # 每次fit前重置分类器
            self.base_classifier.fit(X_balanced, y_balanced)
            X_augmented, y_augmented = X_balanced, y_balanced

        if self.use_difficulty_filtering and self.difficulty_thresholds:
            # Step 3: Filter difficult samples if thresholds are provided
            X_augmented, y_augmented = self._filter_difficult_samples(X_augmented, y_augmented,
                                                                      self.difficulty_thresholds)

        self.base_classifier = copy.deepcopy(self.base_classifier_template)  # 每次fit前重置分类器
        self.base_classifier.fit(X_augmented, y_augmented)

        if self.use_pseudo_labeling:
            # Step 4: Enhance training set and update model with pseudo labels from unlabeled data
            self.base_classifier, X_augmented, y_augmented, _, _ = self._enhance_training_set_and_update_model(
                X_augmented, y_augmented, X_unlabeled, self.pseudo_label, self.pseudo_label_threshold
            )

        return self

    def predict(self, X):
        return self.base_classifier.predict(X)

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)

    def _balance_data_with_smote(self, X, y):
        smote_methods = {'regular': SMOTE, 'borderline': BorderlineSMOTE, 'svm': SVMSMOTE, 'adasyn': ADASYN}
        smote = smote_methods[self.method](random_state=self.random_state)
        return smote.fit_resample(X, y)

    def _generate_pseudo_labels(self, X_unlabeled, X_train, y_train, pseudo_label):
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        pseudo_labeled_data = pd.DataFrame()

        for train_index, test_index in kf.split(X_unlabeled):
            X_train_fold, X_test_fold = X_unlabeled.iloc[train_index], X_unlabeled.iloc[test_index]
            # 这里使用原始训练集和部分未标记数据进行模型训练
            X_combined = pd.concat([X_train, X_train_fold])
            y_combined = pd.concat([y_train, pd.Series([pseudo_label] * len(X_train_fold))])
            self.base_classifier = copy.deepcopy(self.base_classifier_template)  # 每次fit前重置分类器
            self.base_classifier.fit(X_combined, y_combined)
            # 预测剩余部分的未标记数据
            pseudo_labels = self.base_classifier.predict(X_test_fold)

            # 只保留那些伪标签不为-1的样本
            filtered_indices = pseudo_labels != pseudo_label
            filtered_X_test_fold = X_test_fold[filtered_indices]
            filtered_pseudo_labels = pseudo_labels[filtered_indices]

            # 创建一个临时DataFrame来存储伪标签和对应的数据，并将其加入最终的DataFrame中
            temp_df = filtered_X_test_fold.copy()
            temp_df['pseudo_label'] = filtered_pseudo_labels
            pseudo_labeled_data = pd.concat([pseudo_labeled_data, temp_df])

        pseudo_labeled_data.reset_index(drop=True, inplace=True)
        return pseudo_labeled_data

    def _filter_difficult_samples(self, X_train, y_train, difficulty_thresholds):
        y_pred_proba = self.base_classifier.predict_proba(X_train)

        # 复制预测结果用于计算第二大概率
        y_pred_copy = y_pred_proba.copy()
        np.put_along_axis(y_pred_copy, np.argmax(y_pred_proba, axis=1)[:, None], 0, axis=1)

        # 计算每个样本最大概率与第二大概率的差异
        diff = np.max(y_pred_proba, axis=1) - np.max(y_pred_copy, axis=1)

        # 初始化一个布尔数组用于标记要保留的样本
        mask = np.ones(len(y_train), dtype=bool)

        for label, threshold in difficulty_thresholds.items():
            # 找出难以分辨的样本索引
            difficult_samples = np.where((y_train == label) & (diff < threshold))[0]
            # 标记这些样本为False，即不保留
            mask[difficult_samples] = False

        # 使用mask过滤X_train和y_train
        X_train_filtered = X_train[mask]
        y_train_filtered = y_train[mask]

        return X_train_filtered, y_train_filtered

    def _enhance_training_set_and_update_model(self, X_train, y_train, X_test, pseudo_labels, pseudo_label_threshold):
        y_pred_proba = self.base_classifier.predict_proba(X_test)
        y_pred_final = [np.argmax(pred) if max(pred) > pseudo_label_threshold else pseudo_labels for pred in
                        y_pred_proba]

        # 将预测结果合并到测试集中
        X_test_copy = X_test.copy()
        X_test_copy.loc[:, 'label'] = y_pred_final

        # 分离标记过的测试数据和未标记的测试数据
        X_test_labeled = X_test_copy[X_test_copy['label'] != pseudo_labels]
        X_test_unlabeled = X_test_copy[X_test_copy['label'] == pseudo_labels].drop('label', axis=1)

        # 将确定的预测结果加入训练集
        X_train_updated = pd.concat([X_train, X_test_labeled.drop('label', axis=1)])
        y_train_updated = pd.concat([y_train, X_test_labeled['label']])

        # 使用更新后的训练集重新训练模型
        self.base_classifier = copy.deepcopy(self.base_classifier_template)  # 每次fit前重置分类器
        self.base_classifier.fit(X_train_updated, y_train_updated)

        return self.base_classifier, X_train_updated, y_train_updated, X_test_labeled, X_test_unlabeled

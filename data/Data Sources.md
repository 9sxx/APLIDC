## 数据来源

1. **IEEE-CIS Fraud Detection**
   - 源网址：https://www.kaggle.com/c/ieee-fraud-detection/overview
   - 源许可证：https://www.kaggle.com/competitions/ieee-fraud-detection/rules
   - 变量：匿名化的产品、卡片、地址、电子邮箱域、设备、交易日期信息。以V、C、D和M为前缀的数字列，其含义对公众隐藏。
   - 诈骗类别：非现场交易诈骗
   - 提供者：[Vesta Corporation](https://www.vesta.io/)
   - 发布日期：2019-10-03
   - 描述：由IEEE计算智能学会准备，这个非现场交易诈骗数据集在IEEE-CIS诈骗检测Kaggle竞赛中发布，由Vesta Corporation提供。原始数据集包含393个特征，基准中减少到67个特征。特征选择是基于Kaggle内部得到高投票的内核。源数据集训练部分的诈骗率为3.5%。我们仅使用训练文件（train transaction和train identity），包含590,540笔交易，在基准中将其分为训练（95%）和测试（5%）部分，基于时间划分。根据竞赛获胜者的Kaggle内核洞察，我们添加了UUID（称为ENTITY_ID），它代表一个指纹，使用卡、地址、时间和D1特征创建。

2. **Credit Card Fraud Detection**
   - 源网址：https://www.kaggle.com/mlg-ulb/creditcardfraud/
   - 源许可证：https://opendatacommons.org/licenses/dbcl/1-0/
   - 变量：PCA变换后的特征、时间、金额（高度不平衡）
   - 诈骗类别：非现场交易诈骗
   - 提供者：[Machine Learning Group - ULB](https://mlg.ulb.ac.be/)
   - 发布日期：2018-03-23
   - 描述：该数据集包含2013年9月欧洲持卡人的匿名化信用卡交易。数据集在2天内包含284,807笔交易中的492笔诈骗。数据仅包含PCA变换后的数值特征，以及未变换的时间和金额。

3. **Fraud ecommerce**
   - 源网址：https://www.kaggle.com/vbinh002/fraud-ecommerce
   - 源许可证：无
   - 变量：包括注册时间、购买时间、购买价值、设备ID、用户ID、浏览器和IP地址。我们添加了一个新特征，用于测量注册和购买之间的时间差，因为账户的年龄通常是诈骗检测中的一个重要变量。
   - 诈骗类别：非现场交易诈骗
   - 提供者：[Binh Vu](https://www.kaggle.com/vbinh002) 
   - 发布日期：2018-12-09
   - 描述：该数据集包含约15万笔电商交易。

4. **Simulated Credit Card Transactions generated using Sparkov**
   - 源网址：https://www.kaggle.com/kartik2112/fraud-detection
   - 源许可证：https://creativecommons.org/publicdomain/zero/1.0/
   - 变量：交易日期、信用卡号、商户、类别、金额、姓名、街道、性别。所有变量都是使用Sparknov工具合成生成的。
   - 诈骗类别：非现场交易诈骗
   - 提供者：[Kartik Shenoy](https://www.kaggle.com/kartik2112)
   - 发布日期：2020-08-05
   - 描述：这是一个模拟的信用卡交易数据集。数据集使用Sparkov数据生成工具生成，我们修改了为Kaggle创建的数据集版本。它涵盖了6个月内1000名客户与800家商户之间的交易。我们直接从源头使用了训练和测试部分，并随机下采样了测试部分。

5. **Twitter Bots Accounts**
   - 源网址：https://www.kaggle.com/code/davidmartngutirrez/bots-accounts-eda/data?select=twitter_human_bots_dataset.csv
   - 源许可证：https://creativecommons.org/publicdomain/zero/1.0/
   - 变量：包括账户创建日期、粉丝和关注数、个人简介、账户年龄、关于头像和账户活动的元数据，以及标签，指示账户是人类还是机器人。
   - 诈骗类别：机器人攻击
   - 提供者：[David Martín Gutiérrez](https://www.kaggle.com/davidmartngutirrez)
   - 发布日期：2020-08-20
   - 描述：该数据集包含37,438行，对应于Twitter的不同用户账户。

6. **Malicious URLs dataset**
   - 源网址：https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
   - 源许可证：https://creativecommons.org/publicdomain/zero/1.0/
   - 变量：Kaggle数据集使用五个不同的来源整理而成，包含url和类型。虽然原始数据集具有多类标签（类型），我们将其转换为二进制标签。
   - 诈骗类别：恶意流量
   - 提供者：[Manu Siddhartha](https://www.kaggle.com/sid321axn) 
   - 发布日期：2021-07-23
   - 描述：Kaggle数据集使用五个不同的来源整理而成，包含url和类型。虽然原始数据集具有多类标签（类型），我们将其转换为二进制标签。源头没有时间戳信息。因此，我们为了一致性生成了一个虚构的时间戳列。

7. **Real / Fake Job Posting Prediction**
   - 源网址：https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
   - 源许可证：https://creativecommons.org/publicdomain/zero/1.0/
   - 变量：标题、位置、部门、公司、薪资范围、要求、描述、福利、远程工作。大部分变量是分类和自由形式文本的性质。
   - 诈骗类别：内容审核
   - 提供者：[Shivam Bansal](https://www.kaggle.com/shivamb) 
   - 发布日期：2020-02-29
   - 描述：这个Kaggle数据集包含18K个职位描述，其中大约800个是假的。数据包括职位的文本信息和元信息。任务是训练分类模型以检测哪些职位发布是欺诈的。

8. **Vehicle Loan Default Prediction**
   - 源网址：https://www.kaggle.com/avikpaul4u/vehicle-loan-default-prediction
   - 源许可证：未知
   - 变量：贷款人信息、贷款信息、信用局数据和历史记录。
   - 诈骗类别：信用风险
   - 提供者：[Avik Paul](https://www.kaggle.com/avikpaul4u) 
   - 发布日期：2019-11-12
   - 描述：这个数据集的任务是确定车贷违约的概率，特别是首个月分期付款的违约风险。它包含233k笔贷款数据，违约率为21.7%。

# 2020招行fintech精英训练营线上数据赛道复盘

半个月时间参加了招行今年的FinTech训练营数据赛道，最终成绩B榜0.77990。排名60，实在是本人水平有限，大佬们勿笑哈。  
复盘下自己的方案吧

## 一、题目介绍
主办方提供了两个数据集（训练数据集和评分数据集），包含用户标签数据、过去60天的交易行为数据、过去30天的APP行为数据。希望参赛选手基于训练数据集，通过有效的特征提取，构建信用违约预测模型，并将模型应用在评分数据集上，输出评分数据集中每个用户的违约概率。评分指标为AUC。
![比赛内容](https://github.com/muli2464/2020_cmb_fintech/blob/master/%E6%AF%94%E8%B5%9B%E5%86%85%E5%AE%B9.png)

## 二、数据预处理
**1) 缺失值填充**  
“\N”填充为缺失值

**2) 字段变量转换**  
对于一些类别类字段，如性别、标识等，使用label encoder编码方式 (自然数编码)  
对于学历、学位、教育程度、婚姻状况等基数较大的类别特征，使用mean encoder编码方式 (平均数编码)

**3）其他**  
交易行为表和APP行为表将时间字段整理，提取出年、日、小时、是否周末等字段

## 三、特征工程
**交易行为表：**  
1.每个id的交易次数、总交易金额、平均每次交易金额、5月和6月的交易次数和交易金额、劳动节的交易次数和交易金额、周末的交易次数和交易金额  
2.每个id下每种交易方向、支付方式、收支分类的交易金额、交易次数

**APP行为表：**  
整合每个id对每个页面的访问次数数据，再做主成分分析(pca)降到2维

**用户信息表：**  
num类特征进行了一些分桶

## 四、模型
尝试过stacking融合方式，将lightgbm和xgboost的预测结果作为特征，输入线性回归模型中，但是效果不好，最终是采纳了lightgbm单模型。  
模型训练用的是8折交叉验证的方式

## 五、总结与反思
这次工作主要的不足就是特征挖掘的不好，没有充分利用时间数据。尤其是用户最后一次的交易信息没有用到。  

第二个不足就是模型融合效果不好，可能是因为stacking用的是线性回归，应该尝试下别的更复杂点的模型，比如lgb等。当时没有考虑到这一点。

下次希望可以做的更好！

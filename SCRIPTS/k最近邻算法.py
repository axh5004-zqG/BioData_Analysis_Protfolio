# ==========================================
# 零、 准备实验器材 (导入各类工具包)
# ==========================================
import numpy as np  # 引入 numpy 库，专门用来处理多维数组和矩阵运算，就像是超级计算器
import matplotlib.pyplot as plt  # 引入画图神器 matplotlib，负责最后把混淆矩阵和 ROC 曲线画出来
from sklearn.datasets import load_iris  # 从 sklearn 的内置题库里，拿出现成的“鸢尾花”数据集
from sklearn.model_selection import train_test_split, GridSearchCV  # 引入“切分数据集”的剪刀，以及“网格搜索”这台自动跑预实验的机器
from sklearn.neighbors import KNeighborsClassifier  # 引入今天的主角：KNN (K最近邻) 算法模型
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  # 引入用来出具“体检报告”和画“混淆矩阵”的评估工具
from sklearn.metrics import roc_curve, auc, RocCurveDisplay  # 引入专门用来计算和绘制 ROC 曲线及 AUC 值的工具

# ==========================================
# 第一阶段：备料与“双盲”分组 (把数据分成平时练习题和期末考试题)
# ==========================================
print("正在加载数据并划分训练集/测试集...")
iris = load_iris()  # 把鸢尾花数据集加载到电脑内存里，赋值给变量 iris
X = iris.data  # X 代表特征数据（比如花瓣长宽）。在医学里，这就相当于病人的各项生化指标
y = iris.target  # y 代表真实标签（花的品种）。在医学里，这就是病人究竟得没得病（阳性/阴性）

# 用 train_test_split 把数据按 8:2 切分。
# test_size=0.2 意思是留 20% 作为考卷 (X_test, y_test)，80% 作为教材 (X_train, y_train)。
# random_state=42 相当于给随机打乱设一个“种子”，保证你下次跑代码时切分的结果和这次一模一样，方便复现实验。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 第二阶段：疯狂预实验，摸索最佳条件 (网格搜索 + 交叉验证)
# ==========================================
print("\n开始进行网格搜索和 5 折交叉验证...")
knn = KNeighborsClassifier()  # 实例化一个初始的、还没经过任何调教的 KNN 模型

# 拿出一个本子（字典），写下你想让机器去尝试的所有配方组合：
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9],  # 尝试找 1个、3个、5个、7个、9个邻居，看哪个效果好
    'weights': ['uniform', 'distance'],  # 尝试“一视同仁(uniform)”和“距离越近权重越大(distance)”两种计票方式
    'p': [1, 2]  # p=1 代表尝试曼哈顿距离(折线)，p=2 代表尝试欧几里得距离(直线)
}

# 召唤 GridSearchCV 机器。把初始模型(knn)和配方本(param_grid)塞进去，并设定 cv=5 (进行 5 折交叉验证的内部模拟考)
gs = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)

# 发出指令：在 80% 的训练集 (X_train, y_train) 上开始疯狂训练和内部考试！
gs.fit(X_train, y_train)

# 考试结束，让机器汇报它找到的“天下第一”配方
print('【机器汇报最优参数】:', gs.best_params_)
# 打印这套最优配方在 5 次内部模拟考里的平均得分
print(f'【内部交叉验证最高评分】: {gs.best_score_:.4f}')

# ==========================================
# 第三阶段：终极考试 (拿从未见过的 20% 测试集来考)
# ==========================================
# 此时的 gs 已经自动装备了刚才找到的最优配方。
# 我们把锁在保险箱里的那 20% 测试题 (X_test) 喂给它，让它做出预测，把答案保存在 y_pred 里。
y_pred = gs.predict(X_test)

# ==========================================
# 第四阶段：出具深度体检报告 (对对答案，看看错在哪)
# ==========================================
print('\n【分类模型终极评估报告】:')
# 调用 classification_report，对比真实答案(y_test)和模型的预测答案(y_pred)，输出包含查准率、查全率的详细报告
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ----------------- 4.1 绘制混淆矩阵 (看清误诊和漏诊) -----------------
print("\n正在绘制混淆矩阵...")
# 计算真实的混淆矩阵数值矩阵
cm = confusion_matrix(y_test, y_pred)
# 把数值矩阵转化为可以可视化的图形对象，并贴上花的英文名标签
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
# 指定用红色调 (cmap=plt.cm.Reds) 画出热力图
cm_display.plot(cmap=plt.cm.Reds)
plt.title("Confusion Matrix - KNN Iris")  # 给图片加个标题
plt.show()  # 正式把图弹出来显示在屏幕上！(注意：程序会在这里暂停，关掉图片后才会继续往下走)

# ----------------- 4.2 绘制 ROC 曲线 (评估临床诊断价值) -----------------
print("\n正在绘制 ROC 曲线示例...")
# 因为原版的鸢尾花是三分类（三种花），画 ROC 有点复杂。为了让你直观学习，这里手动捏造了一组二分类（阴性/阳性）的病人预测数据
y_test_ex = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1])  # 这是 10 个病人的真实情况（1代表得病，0代表健康）
y_pred_ex = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 0])  # 这是试剂盒（模型）给出的预测结果

# 调用 roc_curve 函数，传入真实值和预测值，它会自动帮你算出 假正例率(fpr) 和 真正例率(tpr)
fpr, tpr, _ = roc_curve(y_test_ex, y_pred_ex)
# 计算这根曲线下的面积，也就是大名鼎鼎的 AUC 值
roc_auc = auc(fpr, tpr)

# 把算出来的 fpr, tpr 和 AUC 值塞进可视化对象里
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
# 画图！
roc_display.plot()
plt.title("ROC Curve Example")  # 给图片加个标题
plt.show()  # 弹出最后一张 ROC 曲线图！大功告成！
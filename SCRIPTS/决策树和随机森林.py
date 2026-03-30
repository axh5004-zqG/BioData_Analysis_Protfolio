# 引入 pandas 数据处理库，它就像是 Python 里的 Excel，专门用来整理数据表格
import pandas as pd
# 引入 matplotlib 画图神器，用来把干巴巴的数字变成高颜值的科研图表
import matplotlib.pyplot as plt
# 从 sklearn 的内置题库里，拿出现成的“鸢尾花”数据集（相当于我们的临床病人数据库）
from sklearn.datasets import load_iris
# 引入数据切分工具，负责把数据集拆分成“平时练习题（训练集）”和“期末考试题（测试集）”
from sklearn.model_selection import train_test_split
# 引入一号选手：决策树分类器（相当于一位单打独斗的专科医生）
from sklearn.tree import DecisionTreeClassifier
# 引入二号选手：随机森林分类器（相当于由众多医生组成的顶级专家会诊团队）
from sklearn.ensemble import RandomForestClassifier
# 引入准确率打分工具，用来给两位医生的最终诊断结果阅卷打分
from sklearn.metrics import accuracy_score

# ==========================================
# 第一阶段：备料与“双盲”数据分组
# ==========================================
print("正在准备病人数据与特征指标...\n")
# 加载鸢尾花数据集，存入变量 iris 中
iris = load_iris()
# 将包含花瓣/花萼长宽的数据（相当于病人的基因表达量）赋值给 X
X = iris.data
# 将真实的花的品种（相当于病人最终的确诊结果）赋值给 y
y = iris.target
# 提取出四个特征的具体名称（比如 'petal length (cm)'），留着后面画图用
feature_names = iris.feature_names

# 用剪刀把数据按 8:2 切分，80% 留给模型学习(Train)，20% 锁起来作为最终盲测卷(Test)
# random_state=42 是为了存档洗牌的顺序，保证每次运行结果完全一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 第二阶段：一号选手（单棵决策树）登场
# ==========================================
print("--- 🩺 一号选手：单棵决策树 ---")
# 实例化决策树模型。限制最大深度 max_depth=3，防止这位医生“死记硬背”导致过拟合
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)

# 让决策树看着 80% 的训练集数据开始学习找规律
dt_model.fit(X_train, y_train)
# 学习完毕，把剩下的 20% 考卷发给它，让它做出预测，答案保存在 dt_pred 里
dt_pred = dt_model.predict(X_test)

# 对对答案，计算这位单打独斗医生的预测准确率
dt_accuracy = accuracy_score(y_test, dt_pred)
# 打印成绩，保留两位小数
print(f"决策树预测准确率: {dt_accuracy * 100:.2f}%\n")

# ==========================================
# 第三阶段：二号选手（随机森林）登场
# ==========================================
print("--- 🏥 二号选手：随机森林联合会诊 ---")
# 实例化随机森林模型。n_estimators=100 代表雇佣 100 位不同的医生（种 100 棵树）
# n_jobs=-1 是解除封印的终极指令：榨干电脑 CPU 的所有核心，全功率并行运算！
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 让 100 位医生开始学习。机器会在后台自动玩“随机抽样”的游戏，给每位医生看不一样的病历
rf_model.fit(X_train, y_train)
# 100 位医生独立看剩下的 20% 考卷，然后通过“少数服从多数”投票，得出最终结论
rf_pred = rf_model.predict(X_test)

# 对对答案，计算顶级会诊团队的预测准确率
rf_accuracy = accuracy_score(y_test, rf_pred)
# 打印团队最终得分
print(f"随机森林预测准确率: {rf_accuracy * 100:.2f}%\n")

# ==========================================
# 第四阶段：杀手锏 —— 提取核心致病标志物排行榜！
# ==========================================
print("--- 🏆 正在提取特征重要性评分 (Biomarker Ranking) ---")

# 从训练好的随机森林模型肚子里，掏出它给所有特征打出的“重要性分数”
importances = rf_model.feature_importances_

# 把特征的名字（列名）和它们对应的分数，打包绑定成一个 pandas 数据表
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
# 对这个数据表按照分数进行排序。ascending=True 表示从小到大排，这样画图时得分最高的在最上面
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=True)

# 召唤画图画板，设置宽 10 英寸，高 6 英寸
plt.figure(figsize=(10, 6))
# 画一张水平方向的柱状图 (bar horizontal)。Y轴是特征名，X轴是分数，涂上好看的珊瑚红色，并加上黑色边框
plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='lightcoral', edgecolor='black')

# 给这幅图表起个高大上的主标题
plt.title("Random Forest Feature Importance (Biomarker Ranking)", fontsize=16)
# 给横坐标（X轴）打上标签，说明这里的横轴代表基尼指数下降的贡献度
plt.xlabel("Importance Score (Gini Decrease)", fontsize=12)
# 给纵坐标（Y轴）打上标签，说明这里列的是具体的特征
plt.ylabel("Features", fontsize=12)
# 在 X 轴方向画上虚线辅助网格，显得图表极具专业科研范儿
plt.grid(axis='x', linestyle='--', alpha=0.7)
# 自动调整排版，防止坐标轴的字被挡住或挤出画面
plt.tight_layout()

# 终极魔法：把这张包含着模型心血的排行榜图表直接弹出显示在屏幕上！
plt.show()




import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
 
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
 
# 生成数据集的GT标签
gt_labels = np.zeros(1000).reshape(10, -1)
for i in range(10):
    gt_labels[i] = i
gt_labels = gt_labels.reshape(1, -1).squeeze()
print("gt_labels.shape : {}".format(gt_labels.shape))
print("gt_labels : {}".format(gt_labels[::5]))
 
# 生成数据集的预测标签
pred_labels = np.zeros(1000).reshape(10, -1)
for i in range(10):
    # 标签生成规则：对于真值类别编号为i的数据，生成的预测类别编号为[0, i-1]之间的随机值
    # 这样生成的预测准确率从0到9逐渐递减
    pred_labels[i] = np.random.randint(0, i + 1, 100)
pred_labels = pred_labels.reshape(1, -1).squeeze()
print("pred_labels.shape : {}".format(pred_labels.shape))
print("pred_labels : {}".format(pred_labels[::5]))
 
# 使用sklearn工具中confusion_matrix方法计算混淆矩阵
confusion_mat = confusion_matrix(gt_labels, pred_labels)
print("confusion_mat.shape : {}".format(confusion_mat.shape))
print("confusion_mat : {}".format(confusion_mat))
 
# 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
disp.plot(
    include_values=True,            # 混淆矩阵每个单元格上显示具体数值
    cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
    ax=None,                        # 同上
    xticks_rotation="horizontal",   # 同上
    values_format="d"               # 显示的数值格式
)
plt.show()
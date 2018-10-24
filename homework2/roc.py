from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
from IPython import embed
y = [1,1,0,0,1,1,0,0,1,0]
y1 = [0.73, 0.69, 0.44, 0.55, 0.67, 0.47, 0.08, 0.15, 0.45, 0.35]
y2 = [0.61, 0.03, 0.68, 0.31, 0.45, 0.09, 0.38, 0.05, 0.01, 0.04]


fpr, tpr, thresholds = roc_curve(y, y1)
fpr2, tpr2, thresholds2 = roc_curve(y, y2)

roc_auc = auc(fpr, tpr)
roc_auc2 = auc(fpr2, tpr2)
embed()
plt.figure()
plt.title('ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, color='darkorange')
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % roc_auc2, color='navy')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


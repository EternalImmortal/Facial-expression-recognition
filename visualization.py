import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

true_private = np.load('total_labels.npy')
predicted_private = np.load('total_predicted.npy')

# sns.set()
# f, ax = plt.subplots()
#
# C2 = confusion_matrix(y_true=true_private, y_pred=predicted_private, labels=[0, 1, 2, 3, 4, 5, 6], normalize='true')
# sns.heatmap(C2, annot=True, ax=ax)
#
# ax.set_title('confusion matrix')  # 标题
# ax.set_xlabel('predict')  # x轴
# ax.set_ylabel('true')  # y轴
#
# plt.show()
# plt.cla()

class_true = list(0. for i in range(7))
class_predicted = list(0. for j in range(7))

for i in range(len(true_private)):
    i = int(i)
    true = int(true_private[i])
    predicted = int(predicted_private[i])
    class_true[true] += 1
    class_predicted[predicted] += 1

counts = np.append(class_true, class_predicted)
counts = pd.DataFrame(data=counts, columns=['count'], dtype=int)

labels = np.append(np.array(emotion_list), np.array(emotion_list))
labels = pd.DataFrame(data=labels, columns=['emotions'])

data_kind_true = np.array(list('real' for i in range(7)))
data_kind_false = np.array(list('predicted' for j in range(7)))
data_kind = np.append(data_kind_true, data_kind_false)
data_kind = pd.DataFrame(data=data_kind, columns=['data_kind'])

matrix = pd.concat([labels, data_kind, counts], axis=1)
print(matrix)
ax = sns.barplot(x='emotions', y='count', hue='data_kind', data=matrix)

plt.show()

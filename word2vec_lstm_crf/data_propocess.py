"""Created by PeterLee, on Dec. 17."""
import re
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score

# tag2label = {"O": 0,
#              "B-PER": 1, "I-PER": 2,
#              "B-LOC": 3, "I-LOC": 4,
#              "B-ORG": 5, "I-ORG": 6
#              }
#
# path_la = r''
# label = '\\label_'
# f1_total = []
# pre_total = []
# recall_total = []
# label_path = path_la + '\\results_all'
# with open(label_path, "w", encoding='utf-8') as fw:
#     for i in range(1, 26):
#         file = path_la + label + str(i)
#         raw_label = []
#         predicted = []
#         with open(file) as f:
#             content = f.readlines()
#             for line in content:
#                 temp = line.strip().split()
#                 if len(temp) != 0:
#                     raw_label.append(int(tag2label[temp[1]]))
#                     predicted.append(int(tag2label[temp[2]]))
#         f1 = f1_score(raw_label, predicted, average='macro')
#         f1_total.append(f1)
#         recall = recall_score(raw_label, predicted, average='macro')
#         recall_total.append(recall)
#         precision = precision_score(raw_label, predicted, average='macro')
#         pre_total.append(precision)
#         results = "Precision = {} , Recall = {}, f1 = {}".format(precision, recall, f1)
#         fw.write(results+'\n')
#
# epoch_len = [i for i in range(len(f1_total))]
# plt.plot(epoch_len, pre_total, color='red', label='Precision')
# plt.plot(epoch_len, recall_total, color='green', label='Recall')
# plt.plot(epoch_len, f1_total, color='blue', label='F1-score')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()

# Path = r''

# with open(Path+'\\train_data.txt', 'w', encoding='utf-8') as f0:
#     with open(Path+'\source_data.txt', encoding='utf-8') as f1, open(Path+'\source_label.txt', encoding='utf-8') as f2:
#         lines_1 = f1.readlines()
#         lines_2 = f2.readlines()
#         for i, j in zip(lines_1, lines_2):
#             for m, n in zip(i.strip().split(), j.strip().split()):
#                 f0.write(m + ' ' + n + '\n')
#             f0.write('\n')


# with open(Path+'\\test_process_data.txt', 'w', encoding='utf-8') as f0:
#     with open(Path+'\\test_data.txt', encoding='utf-8') as f1, open(Path+'\\test_label.txt', encoding='utf-8') as f2:
#         lines_1 = f1.readlines()
#         lines_2 = f2.readlines()
#         for i, j in zip(lines_1, lines_2):
#             for m, n in zip(i.strip().split(), j.strip().split()):
#                 f0.write(m + ' ' + n + '\n')
#             f0.write('\n')




# Path_results = r''
# file_name = '\\result_metric_'
# precision = []
# recall = []
# f1 = []
# for i in range(1, 26):
#     file = Path_results + file_name + str(i)
#     with open(file, encoding='utf-8') as f:
#         content = f.readlines()
#         print(content)
#         result = re.findall(r"\d+\.?\d*", content[0])
#         precision.append(round(float(result[0]), 4))
#         recall.append(round(float(result[1]), 4))
#         f1.append(round(float(result[3]), 4))
# epoch_len = [i for i in range(len(f1))]
# plt.plot(epoch_len, precision, color='red', label='Precision')
# plt.plot(epoch_len, recall, color='green', label='Recall')
# plt.plot(epoch_len, f1, color='blue', label='F1-score')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()




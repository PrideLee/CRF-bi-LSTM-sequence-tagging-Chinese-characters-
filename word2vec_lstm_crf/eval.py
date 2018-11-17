"""Created by PeterLee, on Dec. 17."""
import os
from sklearn.metrics import f1_score, recall_score, precision_score

tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }

def conlleval(label_predict, label_path, metric_path):
    """
    :param label_predict
    :param label_path
    :param metric_path
    :return:
    """
    eval_perl = "./conlleval_rev.pl"
    # Writing the final results
    with open(label_path, "w") as fw:
        line = []
        real_result = []
        predict_result = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag_ = 'O' if tag_ == 0 else tag_
                tag = 'O' if tag == 0 else tag
                line.append("{} {} {}\n".format(char, tag, tag_))
                real_result.append(int(tag2label[tag]))
                predict_result.append(int(tag2label[tag_]))
            line.append("\n")
        f1 = f1_score(real_result, predict_result, average='macro')
        recall = recall_score(real_result, predict_result, average='macro')
        precision = precision_score(real_result, predict_result, average='macro')
        results = "Precision = {} , Recall = {}, f1 = {}".format(precision, recall, f1)
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path, 'w', encoding='utf-8') as fr:
        fr.write(results)
    return results




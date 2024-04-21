import argparse
import pickle

import numpy as np
from tqdm import tqdm

label = open("./val_label.pkl", "rb")
file = pickle.load(label)
_, label_list = file
num_class = int(max(label_list)) + 1
label = np.array(file)
r1 = open("./best_acc_joint.pkl", "rb")
r1 = list(pickle.load(r1).items())
r2 = open("./best_acc_bone.pkl", "rb")
r2 = list(pickle.load(r2).items())
r3 = open("./best_acc_joint_motion.pkl", "rb")
r3 = list(pickle.load(r3).items())
r4 = open("./best_acc_bone_motion.pkl", "rb")
r4 = list(pickle.load(r4).items())

alpha = [1.0, 0.9, 0.5, 0.5]  # 51.50

mean = 0
max_acc = 0
max_acc_5 = 0
max_acc_per_class = 0
max_acc_5_per_class = 0
best_alpha = []
for k1 in np.linspace(1.0, 2.0, 11):
    for k2 in np.linspace(0.5, 1.5, 11):
        for k3 in np.linspace(0.1, 1.0, 10):
            for k4 in np.linspace(0.1, 1.0, 10):
                right_num = total_num = right_num_5 = 0
                names = []
                preds = []
                scores = []
                for i in range(len(label[0])):
                    name, l = label[:, i]
                    # names.append(name)
                    name1, r11 = r1[i]
                    name2, r22 = r2[i]
                    name3, r33 = r3[i]
                    name4, r44 = r4[i]
                    assert name == name1 == name2 == name3 == name4
                    # mean += r11.mean()
                    score = (r11 * k1 + r22 * k2 + r33 * k3 + r44 * k4) / (
                        k1 + k2 + k3 + k4
                    )
                    rank_5 = score.argsort()[-5:]
                    right_num_5 += int(int(l) in rank_5)
                    r = np.argmax(score)
                    scores.append(score)
                    # preds.append(r)
                    right_num += int(r == int(l))
                    total_num += 1
                    # f.write('{}, {}\n'.format(name, r))
                scores = np.stack(scores)
                rank = scores.argsort()

                hit_top_1 = [l in rank[i, -1:] for i, l in enumerate(label_list)]
                acc = [0 for c in range(num_class)]
                for c in range(num_class):
                    hit_label = [l == c for l in label_list]
                    acc[c] = np.sum(
                        np.array(hit_top_1).astype(np.float)
                        * np.array(hit_label).astype(np.float)
                    ) / label_list.count(c)
                acc_per_class = np.mean(acc)

                hit_top_5 = [l in rank[i, -5:] for i, l in enumerate(label_list)]
                acc = [0 for c in range(num_class)]
                for c in range(num_class):
                    hit_label = [l == c for l in label_list]
                    acc[c] = np.sum(
                        np.array(hit_top_5).astype(np.float)
                        * np.array(hit_label).astype(np.float)
                    ) / label_list.count(c)
                acc_per_class_5 = np.mean(acc)

                acc = right_num / total_num
                acc5 = right_num_5 / total_num
                if acc > max_acc:
                    max_acc = acc
                    max_acc_5 = acc5
                    max_acc_per_class = acc_per_class
                    max_acc_5_per_class = acc_per_class_5
                    best_alpha = [k1, k2, k3, k4]
# print(total_num)
print("top1: ", max_acc)
print("top1 per class: ", max_acc_per_class)
print("top5: ", max_acc_5)
print("top5 per class: ", max_acc_5_per_class)
print("best_alpha: ", best_alpha)

# f.close()
# print(mean/len(label[0]))

# with open('./val_pred.pkl', 'wb') as f:
#     # score_dict = dict(zip(names, preds))
#     score_dict = (names, preds)
#     pickle.dump(score_dict, f)

"""with open('./gcn_ensembled.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)"""

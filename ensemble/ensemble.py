import argparse
import pickle

import numpy as np
from tqdm import tqdm

label = open('./val_label.pkl', 'rb')
file = pickle.load(label)
_, label_list = file
num_class = int(max(label_list))+1
label = np.array(file)
r1 = open('./best_acc_joint.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./best_acc_bone.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('./best_acc_joint_motion.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('./best_acc_bone_motion.pkl', 'rb')
r4 = list(pickle.load(r4).items())

alpha = [1.7,0.8,0.5,0.5] # 53.68 for WLASL2000  
#alpha = [1.5,0.6,0.5,0.1]  # WLASL1000
#alpha = [1.5,0.7,0.4,0.4]  # WLASL300
#alpha = [1.7,1.2,0.2,0.3]  # WLASL100
#alpha = [1.3,1.3,0.8,1.0]  # SLR500
#alpha = [1.8,0.5,0.1,0.7]  # AUTSL
#alpha = [1.5,0.6,0.3,0.8]  # MSASL1000
#alpha = [1.6,1.2,0.6,0.9]  # MSASL500
#alpha = [1.7,0.5,0.4,0.6]  # MSASL200
#alpha = [1.9,1.4,0.3,0.4]  # MSASL100


right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0

with open('predictions.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        assert name == name1 == name2 == name3 == name4
        mean += r11.mean()
        score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).sum()
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(score)
        scores.append(score)
        #preds.append(r)
        right_num += int(r == int(l))
        total_num += 1
        f.write('{}, {}\n'.format(name, r))
    scores = np.stack(scores)
    rank = scores.argsort()

    hit_top_1 = [l in rank[i, -1:] for i, l in enumerate(label_list)]
    acc = [0 for c in range(num_class)]
    for c in range(num_class):
        hit_label = [l==c for l in label_list]
        acc[c] = np.sum(np.array(hit_top_1).astype(np.float) * np.array(hit_label).astype(np.float)) / label_list.count(c)
    acc_per_class = np.mean(acc)

    hit_top_5 = [l in rank[i, -5:] for i, l in enumerate(label_list)]
    acc = [0 for c in range(num_class)]
    for c in range(num_class):
        hit_label = [l==c for l in label_list]
        acc[c] = np.sum(np.array(hit_top_5).astype(np.float) * np.array(hit_label).astype(np.float)) / label_list.count(c)
    acc_per_class_5 = np.mean(acc)

    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print('top1: ', acc)
    print('top1 per class: ', acc_per_class)
    print('top5: ', acc5)
    print('top5 per class: ', acc_per_class_5)

f.close()
print(mean/len(label[0]))

# with open('./val_pred.pkl', 'wb') as f:
#     # score_dict = dict(zip(names, preds))
#     score_dict = (names, preds)
#     pickle.dump(score_dict, f)

with open('./gcn_ensembled.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)
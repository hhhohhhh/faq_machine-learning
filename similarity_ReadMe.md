# 实验： 

实验|  思路 |数据集|模型|loss|评估指标| optimizer & learning_rate & stragety |实验效果|
----|-------|------|----|----|-------|---------------------------|---------|
 1  | 人脸识别| 蚂蚁金服的数据集 atec_nlp_sim_train.csv | Train: siamese triplet network, Infer: cosine_similarity(u,v) | online_hard_triplet_loss | recall@100|Adam & 0.005 & None | 结果基本接近 0%，最高 0.12% |
 2  | 人脸识别| atec_nlp_sim_train.csv | Train: siamese triplet network, Infer: cosine_similarity(u,v) | online_hard_triplet_loss | recall@100|Adam & 2e-5 & None | 结果基本接近 0% |
 2  | sbert   | atec_nlp_sim_train.csv | Train: siamese network   +  concate(u,v,u-v) , Infer: cosine_similarity(u,v) | cross-entropy loss|  recall@100 | Adam & 2e-5 & warmup|     |

import pandas as pd
import requests
import json
from tqdm import tqdm
# import xgboost as xgb
import numpy as np
import jieba
import os


def drop_duplicate_question(data):
    data_df = data.copy()
    data_df = data_df[data_df.scenario=='gy']
    data_df.drop_duplicates(subset=["question_id"], inplace=True)
    question = data_df.question.apply(lambda x:"".join(str(x).upper().strip().split()))
    data_df.loc[:,'question'] = question
    data_df.drop_duplicates(subset=['question'],inplace=True)
    print(f"原始的数据共 {data.shape},去除重复后的数据据 共 {data_df.shape}")
    return data_df

def upload(path_std_q, path_ext_q=None, address="192.168.73.51", port=20028):
    url = "http://" + address + ":" + str(port) + "/hsnlp/qa/sentence_similarity/push_data"
    print(f'Upload data to {url}')
    std_df = pd.read_excel(path_std_q, sheet_name="hsnlp_standard_question")
    print(f"读取 标注问数据 {std_df.shape}")
    # 去除重复的数据
    std_df = drop_duplicate_question(std_df)

    question_id_std = std_df["question_id"].tolist()
    question_std = std_df["question"].tolist()
    question_id_set = set(question_id_std)
    question_std_set= set(question_std)

    assert question_id_std.__len__() == question_id_set.__len__()
    assert question_std.__len__() ==  question_std_set.__len__()
    # 推送标准问
    print("开始组装标准问")
    data = []
    for i in range(len(std_df)):
        d = dict()
        d["question_id"] = question_id_std[i]
        d["question"] = "".join(str(question_std[i]).strip().split())
        d["is_standard"] = "1"
        d["standard_id"] = question_id_std[i]
        data.append(d)
        print(f"共组装标准问= {data.__len__()} 条")
    if path_ext_q is not None:
        ext_df = pd.read_excel(path_ext_q, sheet_name="hsnlp_extend_question")
        # data_df = pd.concat([data_df,ext_df])
        print(f"读取 扩展问数据 {ext_df.shape}")
        # 去除重复的数据
        ext_df = drop_duplicate_question(ext_df)

        question_id_ext = ext_df["question_id"].tolist()
        question_ext = ext_df["question"].tolist()
        standard_id = ext_df["standard_id"].tolist()
        print("开始组装扩展问")
        for i in range(len(ext_df)):
            d = dict()
            question_id = question_id_ext[i]
            question = question_ext[i]
            if question_id in question_id_set:
                print(f"扩展问中 question_id={question_id},question={question} question_id 已经在标准问的 question_id_set 中")
                continue
            elif question in question_std_set:
                print(f"扩展问中 question_id={question_id},question={question} question 已经在标准问的 question_id_set 中")
            else:
                d["question_id"] = question_id_ext[i]
                d["question"] = question_ext[i]
                d["is_standard"] = "0"
                d["standard_id"] = standard_id[i]
                data.append(d)
        print(f"共组装标准问+扩展问= {data.__len__()} 条")
    cont = requests.post(url, data={"data": json.dumps(data)}).content
    return_content = json.loads(cont)
    print(f"返回内容： {return_content}")
    return return_content


def get(q: str, address="192.168.73.51", port=20028, threshold=0.0):
    assert q != ""
    # print(q)
    url_sim = "http://" + address + ":" + str(port) + "/hsnlp/qa/sentence_similarity/similarity"
    content = requests.post(url_sim, data={"query": q, "threshold": threshold}).content
    # print(content)
    result = json.loads(content)["result"]
    return result


def evaluation_accuracy(path_input, path_output, fn_threshold=None, sheet_name="Sheet1", threshold=0.8):
    df = pd.read_excel(path_input, sheet_name=sheet_name)
    # df = df[:10]
    if "标准问" not in df.columns:
        df["标准问"] = "na"
    list_eval_q = []
    list_eval_sq = []
    list_retrieved_q = []
    list_retrieved_sq = []
    list_retrieved_score = []
    list_retrieved_logit = []
    list_retrieved_recall_score = []
    list_rank = []
    list_result = []
    count_total = 0
    count_na = 0
    count_na_right = 0
    count_right = 0
    for index, row in tqdm(df.iterrows()):
        count_total += 1
        result = get(row["评估问"], port=PORT)

        # 无召回
        if not result["recall_question"]:

            list_eval_q.append(row["评估问"])
            list_eval_sq.append(row["标准问"])
            list_retrieved_q.append("空")
            list_retrieved_sq.append("空")
            list_retrieved_score.append(0.0)
            list_retrieved_logit.append(0.0)
            list_retrieved_recall_score.append(0.0)
            list_rank.append(-1)

            if row["标准问"] == "na":
                count_na += 1
                count_na_right += 1
                list_result.append("正确-无对应标准问召回空")
            else:
                list_result.append("错误-有对应标准问召回空")
            continue

        # print(result)

        # 有召回
        list_eval_q.extend([row["评估问"] for _ in result["recall_question"]])
        list_eval_sq.extend([row["标准问"] for _ in result["recall_question"]])
        list_retrieved_q.extend(result["recall_question"])
        list_retrieved_sq.extend(result["recall_std_question"])
        list_retrieved_score.extend(result["recall_score"])
        # list_retrieved_logit.extend(result["logits"])
        list_retrieved_recall_score.extend(result["bm25_score"])
        list_rank.extend([_ for _ in range(1, len(result["recall_question"])+1)])

        if fn_threshold is not None:
            # input feature
            input_data = xgb.DMatrix(np.array([[result["recall_bm25_score"][0],
                                                len(row["评估问"].strip("？")),
                                                len(result["recall_question"][0].strip("？")),
                                                len(list(jieba.cut(row["评估问"].strip("？")))),
                                                len(list(jieba.cut(result["recall_question"][0].strip("？"))))
                                                ]]))
            threshold_predict = fn_threshold.predict(input_data)[0]
        else:
            threshold_predict = result["recall_score"][0]

        # 无对应标准问不应命中
        if row["标准问"] == "na":
            count_na += 1
            if threshold_predict <= threshold:
                count_na_right += 1
                list_result.extend(["正确-无对应标准问不超过阈值不命中" for _ in result["recall_question"]])
            else:
                list_result.extend(["错误-无对应标准问但超过阈值命中" for _ in result["recall_question"]])
        # 有对应标准问应命中
        else:
            if row["标准问"] == result["recall_std_question"][0]:
                if threshold_predict > threshold:
                    count_right += 1
                    list_result.extend(["正确-有对应标准问有相关召回命中正确" for _ in result["recall_question"]])
                else:
                    list_result.extend(["错误-有对应标准问有相关召回命中正确阈值不达到" for _ in result["recall_question"]])
            elif row["标准问"] in result["recall_std_question"]:
                if threshold_predict > threshold:
                    list_result.extend(["错误-有对应标准问有相关召回命中错误" for _ in result["recall_question"]])
                else:
                    list_result.extend(["错误-有对应标准问有相关召回命中错误阈值不达到" for _ in result["recall_question"]])
            else:
                if threshold_predict > threshold:
                    list_result.extend(["错误-有对应标准问没有相关召回命中错误" for _ in result["recall_question"]])
                else:
                    list_result.extend(["错误-有对应标准问没有相关召回命中错误阈值不达到" for _ in result["recall_question"]])

    print("评估问数量", count_total, "总体准确率", (count_right + count_na_right) / count_total)
    if count_na > 0:
        print("其中无对应标准问评估问数量", count_na, "无对应标准问不应命中准确率", count_na_right / count_na)
    if count_total > count_na:
        print("其中有对应标准问评估问数量", count_total - count_na, "有对应标准问top1准确率",
              count_right / (count_total - count_na))

    df_output = pd.DataFrame({"评估问": list_eval_q, "标准问": list_eval_sq, "召回扩展问": list_retrieved_q,
                              "召回标准问": list_retrieved_sq, "相似度": list_retrieved_score,
                             # "召回bm25分数": list_retrieved_recall_score, "rank": list_rank, "结果": list_result})
                             "rank": list_rank, "结果": list_result})
    df_output.to_excel(path_output, header=True, index=False)
    print("结果写入文件", path_output)


def evaluation_zhengfanchangjin(path_input, path_output, fn_threshold=None, sheet_name="Sheet1", threshold=0.8):

    def _is_exist(line):
        if row["评估问所属场景"] == "正向场景":
            return 1 if line["正向场景是否存在"] == "是" else 0
        elif row["评估问所属场景"] == "反向场景":
            return 1 if line["反向场景是否存在"] == "是" else 0
        raise KeyError("评估问所属场景与正反向场景均不存在")

    df = pd.read_excel(path_input, sheet_name=sheet_name)
    assert "标准问" in df.columns
    list_eval_q = []
    list_eval_sq = []
    list_retrieved_q = []
    list_retrieved_sq = []
    list_retrieved_score = []
    list_retrieved_logit = []
    list_retrieved_recall_score = []
    list_rank = []
    list_result = []
    count_total_1 = 0
    count_total_2 = 0
    count_right_cond_1 = 0
    count_right_cond_2 = 0
    for index, row in tqdm(df.iterrows()):
        assert row["标准问"] != "na"
        result = get(row["评估问"], port=PORT)

        cond_1_1 = row["正向场景是否存在"] == "是" and row["反向场景是否存在"] == "是"
        cond_1_2 = (not (row["正向场景是否存在"] == "否" and row["反向场景是否存在"] == "否")) and (_is_exist(row) == 1)
        cond_2 = (not (row["正向场景是否存在"] == "否" and row["反向场景是否存在"] == "否")) and (_is_exist(row) == 0)
        if cond_1_1 or cond_1_2:  # 常规流程
            pass
        if cond_2:  # top1不为对向场景标准问即可
            pass

        # 无召回
        if not result["recall_question"]:

            list_eval_q.append(row["评估问"])
            list_eval_sq.append(row["标准问"])
            list_retrieved_q.append("空")
            list_retrieved_sq.append("空")
            list_retrieved_score.append(0.0)
            list_retrieved_logit.append(0.0)
            list_retrieved_recall_score.append(0.0)
            list_rank.append(-1)

            if cond_1_1 or cond_1_2:
                count_total_1 += 1
                list_result.append("错误-有对应标准问召回空")
            elif cond_2:
                count_total_2 += 1
                count_right_cond_2 += 1
                list_result.append("正确-无对应标准问召回空")
            else:
                list_result.append("双否")
            continue

        # 有召回
        list_eval_q.extend([row["评估问"] for _ in result["recall_question"]])
        list_eval_sq.extend([row["标准问"] for _ in result["recall_question"]])
        list_retrieved_q.extend(result["recall_question"])
        list_retrieved_sq.extend(result["recall_std_question"])
        list_retrieved_score.extend(result["recall_score"])
        list_retrieved_logit.extend(result["logits"])
        list_retrieved_recall_score.extend(result["bm25_score"])
        list_rank.extend([_ for _ in range(1, len(result["recall_question"])+1)])

        if fn_threshold is not None:
            # input feature
            input_data = xgb.DMatrix(np.array([[result["recall_bm25_score"][0],
                                                len(row["评估问"].strip("？")),
                                                len(result["recall_question"][0].strip("？")),
                                                len(list(jieba.cut(row["评估问"].strip("？")))),
                                                len(list(jieba.cut(result["recall_question"][0].strip("？"))))
                                                ]]))

            threshold_predict = fn_threshold.predict(input_data)[0]
        else:
            threshold_predict = result["recall_score"][0]

        # 有对应标准问应命中
        if cond_1_1 or cond_1_2:
            count_total_1 += 1
            if row["标准问"] == result["recall_std_question"][0]:
                if threshold_predict > threshold:
                    count_right_cond_1 += 1
                    list_result.extend(["正确-有对应标准问有相关召回命中正确" for _ in result["recall_question"]])
                else:
                    list_result.extend(["错误-有对应标准问有相关召回命中正确阈值不达到" for _ in result["recall_question"]])
            elif row["标准问"] in result["recall_std_question"]:
                if threshold_predict > threshold:
                    list_result.extend(["错误-有对应标准问有相关召回命中错误" for _ in result["recall_question"]])
                else:
                    list_result.extend(["错误-有对应标准问有相关召回命中错误阈值不达到" for _ in result["recall_question"]])
            else:
                if threshold_predict > threshold:
                    list_result.extend(["错误-有对应标准问没有相关召回命中错误" for _ in result["recall_question"]])
                else:
                    list_result.extend(["错误-有对应标准问没有相关召回命中错误阈值不达到" for _ in result["recall_question"]])
        elif cond_2:
            count_total_2 += 1
            if row["标准问"] == result["recall_std_question"][0]:
                if threshold_predict > threshold:
                    list_result.extend(["错误-命中反向标准问" for _ in result["recall_question"]])
                else:
                    count_right_cond_2 += 1
                    list_result.extend(["正确-未命中反向标准问且top1" for _ in result["recall_question"]])
                continue
            count_right_cond_1 += 1
            list_result.extend(["正确-未命中反向标准问且非top1" for _ in result["recall_question"]])
        else:
            list_result.extend(["双否" for _ in result["recall_question"]])

    print("存在场景评估问", count_total_1, "总体准确率", count_right_cond_1 / count_total_1)
    print("不存在场景评估问", count_total_2, "总体准确率", count_right_cond_2 / count_total_2)

    df_output = pd.DataFrame({"评估问": list_eval_q, "标准问": list_eval_sq, "召回扩展问": list_retrieved_q,
                              "召回标准问": list_retrieved_sq, "相似度": list_retrieved_score, "logits": list_retrieved_logit,
                             "召回bm25分数": list_retrieved_recall_score, "rank": list_rank, "结果": list_result})
    df_output.to_excel(path_output, header=True, index=False)
    print("结果写入文件", path_output)


def evaluation_recall(path_input, path_output, sheet_name="Sheet1"):
    df = pd.read_excel(path_input, sheet_name=sheet_name)
    assert "标准问" in df.columns

    list_eval_q = []
    list_eval_sq = []
    list_retrieved_q = []
    list_retrieved_sq = []
    list_retrieved_score = []
    list_result = []
    list_rank = []
    count_total = 0
    count_recall = 0
    count_recall_20 = 0
    count_recall_10 = 0
    count_recall_5 = 0

    for index, row in tqdm(df.iterrows()):
        count_total += 1
        result = get(row["评估问"], port=PORT)

        # 无召回
        if not result["bm25_recall"]:
            print("无召回结果", row["评估问"])
            list_eval_q.append(row["评估问"])
            list_eval_sq.append(row["标准问"])
            list_retrieved_q.append("空")
            list_retrieved_sq.append("空")
            list_retrieved_score.append(0.0)
            list_result.append("无召回")
            list_rank.append(-1)
            continue

        # 有召回
        list_eval_q.extend([row["评估问"] for _ in result["bm25_recall"]])
        list_eval_sq.extend([row["标准问"] for _ in result["bm25_recall"]])
        list_retrieved_q.extend(result["bm25_recall"])
        list_retrieved_sq.extend(result["bm25_recall_standard"])
        list_retrieved_score.extend(result["bm25_score"])
        list_rank.extend([_ for _ in range(1, len(result["bm25_recall"])+1)])

        if row["标准问"] in result["bm25_recall_standard"]:
            count_recall += 1
            list_result.extend(["召回" for _ in result["bm25_recall"]])
        else:
            list_result.extend(["不召回" for _ in result["bm25_recall"]])

        if row["标准问"] in result["bm25_recall_standard"][:5]:
            count_recall_5 += 1
            count_recall_10 += 1
            count_recall_20 += 1
        elif row["标准问"] in result["bm25_recall_standard"][5:10]:
            count_recall_10 += 1
            count_recall_20 += 1
        elif row["标准问"] in result["bm25_recall_standard"][10:20]:
            count_recall_20 += 1

    print("总体召回率-top30", count_recall / count_total)
    print("总体召回率-top20", count_recall_20 / count_total)
    print("总体召回率-top10", count_recall_10 / count_total)
    print("总体召回率-top5", count_recall_5 / count_total)

    df_output = pd.DataFrame({"评估问": list_eval_q, "标准问": list_eval_sq, "召回扩展问": list_retrieved_q,
                              "召回标准问": list_retrieved_sq, "召回相似度": list_retrieved_score, "rank": list_rank, "结果": list_result})
    df_output.to_excel(path_output, header=True, index=False)
    print("结果写入文件", path_output)


def evaluation_duozhuti(path_input, path_output, sheet_name="Sheet1"):
    df = pd.read_excel(path_input, sheet_name=sheet_name)
    assert "标准问1" in df.columns and "标准问2" in df.columns
    list_eval_q = []
    list_eval_sq1 = []
    list_eval_sq2 = []
    list_retrieved_q = []
    list_retrieved_sq = []
    list_retrieved_score = []
    list_result = []
    count_total = 0
    count_right = 0
    for index, row in tqdm(df.iterrows()):
        count_total += 1
        result = get(row["评估问"], port=PORT)

        # 无召回
        if not result["recall_question"]:

            list_eval_q.append(row["评估问"])
            list_eval_sq1.append(row["标准问1"])
            list_eval_sq2.append(row["标准问2"])
            list_retrieved_q.append("空")
            list_retrieved_sq.append("空")
            list_retrieved_score.append(0.0)

            list_result.append("错误-无召回")
            continue

        # 有召回
        list_eval_q.extend([row["评估问"] for _ in result["recall_question"]])
        list_eval_sq1.extend([row["标准问1"] for _ in result["recall_question"]])
        list_eval_sq2.extend([row["标准问2"] for _ in result["recall_question"]])
        list_retrieved_q.extend(result["recall_question"])
        list_retrieved_sq.extend(result["recall_std_question"])
        list_retrieved_score.extend(result["recall_score"])

        # 有对应标准问应命中
        if row["标准问1"] in result["recall_std_question"][:2] and row["标准问2"] in result["recall_std_question"][:2]:
            count_right += 1
            list_result.extend(["正确-标准问排名前二" for _ in result["recall_question"]])
        else:
            list_result.extend(["错误-标准问未排名前二" for _ in result["recall_question"]])

    print("评估问数量", count_total, "总体准确率", count_right / count_total)

    df_output = pd.DataFrame({"评估问": list_eval_q, "标准问1": list_eval_sq1, "标准问2": list_eval_sq2,
                              "召回扩展问": list_retrieved_q, "召回标准问": list_retrieved_sq, "相似度": list_retrieved_score,
                              "结果": list_result})
    df_output.to_excel(path_output, header=True, index=False)
    print("结果写入文件", path_output)


def main_evaluation(dir_input, dir_output, fn_threshold=None, threshold=0.8):
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
        print("输出路径不存在，创建输出路径", dir_output)
    if fn_threshold is not None:
        print("使用预测分类阈值")
    print("命中阈值", threshold)

    # list_sheet_names = ["合并筛选评估问", "定义类有标准问", "定义类无标准问"]
    # list_dataset_names = ["1.1-合并总体评估集", "1.1-定义类有标准问评估集", "1.1-定义类无标准问评估集"]
    # for sheet_name, dataset_name in zip(list_sheet_names, list_dataset_names):
    #     # print("当前评估集", dataset_name + "召回")
    #     # evaluation_recall(path_input=os.path.join(dir_input, "1.1评估集.xlsx"),
    #     #                   path_output=os.path.join(dir_output, dataset_name + "召回集.xlsx"),
    #     #                   sheet_name=sheet_name)
    #     # print("\n")
    #
    #     print("当前评估集", dataset_name + "top1准确率")
    #     evaluation_accuracy(
    #         path_input=os.path.join(dir_input, "1.1评估集.xlsx"),
    #         path_output=os.path.join(dir_output, dataset_name + "精排集.xlsx"),
    #         sheet_name=sheet_name,
    #         threshold=threshold,
    #         fn_threshold=fn_threshold)
    #     print("\n")

    list_sheet_names = ["1.2-答非所问-金融语境不相关问题", "1.2-答非所问-闲聊语境不相关问题", "1.2-答非所问-短语类问题-有标准问",
                        "1.2-答非所问-短语类问题-无标准问", "1.2-答非所问-单个词", "1.3寒暄评估问", "1.4反向词汇", "1.4反向场景",
                        "1.5增删词该不改变原意", "2.1语气词", "2.1语气词评估问", "2.2标点符号", "2.2标点符号评估问",
                        "2.3其它行业问题", "2.3法律行业词汇", "2.3法律行业词汇评估问", "2.4新兴词汇", "2.4新兴词汇问题",
                        "3.1增删从句评估问"]

    # list_sheet_names = ["1.2-答非所问-短语类问题-有标准问",
    #                     "1.5增删词该不改变原意", "2.1语气词评估问", "2.2标点符号评估问"]
    # list_sheet_names = ["1"]

    for sheet_name in list_sheet_names:
        # print("当前评估集", sheet_name)
        evaluation_accuracy(
            path_input=os.path.join(dir_input, "其它评估集.xlsx"),
            path_output=os.path.join(dir_output, sheet_name + ".xlsx"),
            sheet_name=sheet_name,
            threshold=threshold,
            fn_threshold=fn_threshold)
        print("\n")

    # evaluation_zhengfanchangjin(path_input=os.path.join(dir_input, "其它评估集.xlsx"),
    #                             path_output=os.path.join(dir_output, "1.4正反向场景评估问" + ".xlsx"),
    #                             threshold=threshold,
    #                             fn_threshold=fn_threshold,
    #                             sheet_name="1.4反向场景")

    evaluation_duozhuti(
        path_input=os.path.join(dir_input, "其它评估集.xlsx"),
        path_output=os.path.join(dir_output, "3.2多主题评估问" + ".xlsx"),
        sheet_name="3.2多主题评估问")

    print("完成")


def evaluation_sun_finance():
    list_sheet_names = ["金融", "股票", "基金"]
    for name in list_sheet_names:
        evaluation_accuracy(path_input=r"./评估集/其它评估集.xlsx",
                            path_output=r"./评估结果_" + name + ".xlsx",
                            sheet_name=name, threshold=0.8)


if __name__ == '__main__':
    PORT = 21327  # 服务端口
    question_dir = os.path.join('data','hsnlp_faq_data','国元')
    standard_question_filename = 'hsnlp_standard_question.xlsx'
    extend_question_filename = "hsnlp_extend_question.xlsx"
    result = upload(path_std_q=os.path.join(question_dir,standard_question_filename),
                    path_ext_q=os.path.join(question_dir,extend_question_filename),
                    port=PORT)

    # 读取 question:

    # result = get(q=question,port=PORT)

    # # 上传
    # result = upload(path_std_q=r"./知识库/hsnlp_standard_question.xlsx",
    #                 path_ext_q=r"./其他/new_hsnlp_extend_question.xlsx",
    #                 # path_ext_q=r"./知识库/hsnlp_extend_question.xlsx",
    #                 # path_ext_q=None,
    #                 port=21328)

    # result= upload(path_std_q=r"./知识库/分类0425/包含/证券/hsnlp_standard_question.xlsx"),
    # result=upload(path_std_q=r"./评估集/开源证券/hsnlp_standard_question.xlsx",
    # result=upload(path_std_q=r"./反向词汇/hsnlp_standard_question.xlsx",
    # result=upload(path_std_q=r"./评估集/首创证券/hsnlp_standard_question_online.xlsx",
    # result=upload(path_std_q=r"./评估集/东方证券/hsnlp_standard_question(9).xlsx",
                  # path_ext_q=r"./评估集/开源证券/hsnlp_extend_question.xlsx",
                  # path_ext_q=None,
                  # port=PORT)
                  # port=22228)
    # print(result)

    # 请求单个问题
    # print(str(get("个股期权", threshold=0.0, port=20039)))

    # 评估主函数
    # main_evaluation(dir_input=r"E:\工作文件\智能问答相关\FAQ\语义算法迭代\评估集和评估脚本\评估集",
    #                 dir_output=r"E:\工作文件\智能问答相关\FAQ\语义算法迭代\评估集和评估脚本\评估集\评估结果_非意图",
    #                 threshold=0.8, fn_threshold=None)

    # 单个评估集评估召回
    # evaluation_recall(path_input=r"C:\Users\hspcadmin\Documents\工作任务\券商faq\优化方案\数据\万联-标准问_评估问整理\标准问集_评估问集_20210303\评估集\1.1评估集.xlsx",
    #                   path_output=r"C:\Users\hspcadmin\Documents\工作任务\券商faq\优化方案\数据\万联-标准问_评估问整理\标准问集_评估问集_20210303\评估集\1.1评估集召回集.xlsx",
    #                   sheet_name="合并筛选评估问")

    # # 单个评估集评估准确率
    # 评估总体和定义类
    # evaluation_accuracy(path_input=r"./评估集/1.1评估集.xlsx",
    #                     path_output=r"./评估集/评估结果/合并筛选评估问_添加扩展问.xlsx",
    #                     sheet_name="1.1-合并总体评估集")
    # evaluation_accuracy(path_input=r"./评估集/1.1评估集.xlsx",
    #                     path_output=r"./评估集/评估结果/定义类有标准问_新模型.xlsx",
    #                     sheet_name="1.1-定义类有标准问评估集")
    #
    # evaluation_accuracy(path_input=r"./评估集/sunhl/新词评估集.xlsx",
    #                     path_output=r"./评估集/sunhl/新词评估结果.xlsx",
    #                     sheet_name="Sheet1")
    # #
    # evaluation_accuracy(path_input=r"./反向词汇/反向词汇评估问.xlsx",
    #                     path_output=r"./反向词汇/反向词汇评估问-java结果.xlsx",
    #                     sheet_name="Sheet1")

    # evaluation_accuracy(path_input=r"./评估集/sun_0326/file_4/评估集.xlsx",
    #                     path_output=r"./评估集/sun_0326/file_4/评估结果" + "新模型" + ".xlsx",
    # #                     sheet_name="Sheet1", threshold=0.8)
    # evaluation_accuracy(path_input=r"./评估集/东方证券/评估问.xlsx",
    # # # evaluation_accuracy(path_input=r"./评估集/首创证券/最新评估集-v4.xlsx",
    #                     path_output=r"./评估集/东方证券/东方证券评估结果.xlsx",
    #                     sheet_name="Sheet3")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@file: event_classifier_request.py
#@version: 02
#@author: $ {USER}
#@contact: wangfc27441@***
#@time: 2019/10/25 11:58 
#@desc:

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "event_classifier"))
import pandas as pd
import json
import requests
from event_classifier.data_explore import save_dataframe_to_raw_data
from event_classifier_server.utils.logging_config import logger
from event_classifier.evaluation import save_predict_results,multilabels_evaluation
import tqdm
from build_event_rule import sub_other_type2same_level_labels_dict,event_type2regular_expression_dict


def event_classifier_request(url, title ,text ):
    """
    @author:wangfc27441
    @desc: predictions 为 [['拖欠工资', 1.0]]
    @version：
    @time:2020/9/23 10:54

    """
    post_data = dict(title=title, content=text)
    response_data = requests.post(url, data=post_data)
    response_dict = json.loads(response_data.text)
    predictions = response_dict['pred']
    # print("The predictions is {}".format(predictions))
    return predictions

    # # 批数据测试：
    # test_path = os.path.join('event_classifier', 'corpus', 'important_events.test')
    # raw_test_df = save_dataframe_to_raw_data(path=test_path, mode='r')
    # head_data = raw_test_df.loc[0:2, ['title', 'text']]
    # test_types = ['对外担保','iso', '员工持股计划']
    # batch_data = [dict(title=title, content=text, test_type=test_type) for (title, text),test_type in zip(head_data.values.tolist(),test_types)]
    #
    # # dict对象 转换为 json对象
    # # json_batch_data = json.dumps(batch_data,ensure_ascii=False)
    # logger.info("batch_data={}".format(batch_data))#,json_batch_data))


    # POST 到服务器出现乱码，通过配置header 设置编码解决
    # headers = {'Content-Type': 'application/json;charset=utf-8'}
    # post方法提交 json 数据,注意 这儿 json 数据直接使用 dict对象传入，而不需要转换为 json 对象
    # response_data = requests.post(url, json=batch_data)
    # response_dict = json.loads(response_data.text)
    # predictions = response_dict['pred']
    # print("The predictions is {}".format(predictions))



def test_request():
    label_ids = 20*[0]
    input_ids = 512*[1]
    input_mask = 512*[1]
    segment_ids = 512*[1]
    data_dict_temp = {
            'label_ids': label_ids,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
    }
    data_list = []
    data_list.append(data_dict_temp)

    data = json.dumps({"signature_name": "serving_default", "instances": data_list})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/cail_elem:predict', data=data, headers=headers)
    print(json_response.text)
    predictions = json.loads(json_response.text)['predictions']
    print(predictions)


def request_from_raw_text():
    """

    :return:
    """
    BERT_VOCAB = "/home/data1/ftpdata/pretrain_models/bert_tensoflow_version/bert-base-chinese-vocab.txt"
    text_list = ["权人宏伟支行及宝成公司共22次向怡天公司催收借款全部本金及利息，均产生诉讼时效中断的法律效力，本案债权未过诉讼时效期间",  # LN8
                 ]
    data_list = []
    tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)
    predict_examples = create_examples_text_list(text_list)
    for (ex_index, example) in enumerate(predict_examples):
        feature = convert_single_example(ex_index, example,
                                         512, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = {}
        features["input_ids"] = feature.input_ids
        features["input_mask"] = feature.input_mask
        # pdb.set_trace()
        features["segment_ids"] = feature.segment_ids
        if isinstance(feature.label_ids, list):
            label_ids = feature.label_ids
        else:
            label_ids = feature.label_ids[0]
        features["label_ids"] = label_ids
        # tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        data_list.append(features)


    data = json.dumps({"signature_name": "serving_default", "instances": data_list})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/cail_elem:predict', data=data, headers=headers)
    # print(json_response.text)
    # pdb.set_trace()
    predictions = json.loads(json_response.text)['predictions']
    # print(predictions)
    for p in range(len(predictions)):
        p_list = predictions[p]
        label_index = np.argmax(p_list)
        print("content={},label={}".format(text_list[p], label_index+1))
    print("total number=", len(text_list))

def evaluation_test_data(url=None,test_data_dir=None,test_data_file=None,top_k =1,test_data=None,test_no=None):
    if test_data is None:
        test_data_path = os.path.join(test_data_dir,test_data_file)
        test_no = int(test_data_file.split('.')[0].split('_')[-1]) +1
        # 读取测试数据
        if test_data_path[-3:] =='csv':
            test_data = save_dataframe_to_raw_data(path=test_data_path,mode='r')
        else:
            test_data = pd.read_excel(test_data_path, encoding='utf_8_sig', index_col=None)
        print("读取测试数据test_data.shape={} from {}".format(test_data.shape, test_data_path))
        sub_other_types = list(sub_other_type2same_level_labels_dict.keys())
        sub_other_types.remove("其他事件类型")
        test_data =test_data[~test_data.event_label.isin(sub_other_types)] #.iloc[:10,:]
        print("读取测试数据test_data.shape={} from {}".format(test_data.shape,test_data_path))

    Y_test = []
    Y_pred = []
    for i in tqdm.trange(len(test_data)):
        row = test_data.iloc[i]
        label = row['event_label']
        title = row['title']
        content = row['text']
        # request 请求得到 prediction
        prediction = event_classifier_request(url,title,content)
        Y_pred.append(prediction)
        Y_test.append(label)
    ### 修改为多标签的评估：Evaluation
    first_true_label_ls, predict_result, tag_list, predict_result_whole, weight_result = multilabels_evaluation(
        Y_test, Y_pred, top_k=top_k,evaluation_result_path = test_data_dir)

    ### Save_predict_results

    predict_results_path = os.path.join(test_data_dir,'feedback_data_predict_results_top_{0}_{1}.xlsx'.format(top_k,test_no))
    save_predict_results(test_data, first_true_label_ls, predict_result, tag_list, predict_result_whole,
                         weight_result, predict_results_path)


def read_model_test_data(test_dir,model_test_result_file='predict_results_top_1.xlsx'):
    """
    @author:wangfc27441
    @desc:  读取模型的测试数据和统计结果
    @version：
    @time:2020/9/27 11:31

    """

    from data_analysis import get_classification_report_top1
    import pandas as pd

    # 读取测试报告
    model_test_report_path = os.path.join(test_dir, 'classification_report_top_1.csv')
    classification_report_top1_df, classification_report_top1_summary_df = get_classification_report_top1(
        model_test_report_path)
    classification_report_top1_df.support = classification_report_top1_df.support.astype(int)
    # 读取测试结果
    model_test_result_path = os.path.join(test_dir, model_test_result_file)
    predict_results_top1_data = pd.read_excel(model_test_result_path, encoding='utf_8_sig', index_col=None)
    return predict_results_top1_data, classification_report_top1_df,classification_report_top1_summary_df


def test_rule_with_data(predict_results_top1_data,classification_report_top1_df):
    classification_report_tail_recall = classification_report_top1_df[classification_report_top1_df.support > 0].sort_values(
        by=['recall',"support"],ascending=[True,False]).head(30)
    classification_report_tail_precision = classification_report_top1_df[classification_report_top1_df.support > 0].sort_values(
        by=['precision',"support"],ascending=[True,False]).head(30)
    print("precision较低，需要处理的类型：\n{}".format(classification_report_tail_precision))
    print("recall较低，需要处理的类型：\n{}".format(classification_report_tail_recall))

    for i in range(0,len(classification_report_tail_f1_score)):
        # 筛选测试数据
        test_label = classification_report_tail_f1_score.iloc[i]['label']
        # test_label ="违法违规其他事件" "资金回收风险" "项目资金未按计划到位" "日常经营其他事件" "经营环境发生变化" "流动性风险" "拖欠款项"
        # "IPO存疑" "控股股东变更"  "环境污染问题" "高管能力不足" "合并、分立"  "业务扩张"  "资产负债率过高" "股权资产其他事件" "财务造假或被质疑"  "债务展期" "不可抗力因素"
        # "债券违约" "金融产品问题" "经营业绩" "违规担保" "主营业务变更" "制假售假"  "股权斗争" "指标接近监管红线"
        #"企业解散" # "高管违法" # "主营业务变更"# "延迟评级" #"环境污染问题" #"盲目扩张" "流动性风险" # "退市风险" # "恶意竞争" # "商业侵权" # "经营业绩" # "担保代偿风险"#"信披违规" # "资金回收风险" # "合并、分立" # "盲目扩张" #  "业务扩张" # "流动性风险" #"退市风险" #"摘星摘帽" #"控股股东变更" #"产品价格变动"  # "融资交易其他事件"  #公司基本信息其他事件
        test_label =  "招投标" #"关闭下属单位"#"对外投资"# "理财投资"#"IPO"#"重大突破"##"项目投资"#"上调股票评级" #"股份转让" # "资产重组失败/取消"#"撤销资质"#"出售资产" #"停复牌" #"企业信息变更" #"退市风险" #"市场竞争力下降" #"重要合作取消" # "撤销失信名单" #"终止上市" #"对外投资"# "生产要素成本变动"# "对外赔付" #"行政处罚" #"合同终止"#"对重要子公司失去控制"#"资产减值" #"增资" #"终止上市" #"发行失败"# "移出观察名单" #"信用评级上调"# "评级关注"#"业务扩张" # "盲目扩张" #"列入观察名单"# "维持信用评级"
        test_data = predict_results_top1_data[(predict_results_top1_data.event_label == test_label)] #& (predict_results_top1_data.tag==False)]
        # test_data = predict_results_top1_data[(predict_results_top1_data.predict_label == test_label) & (predict_results_top1_data.tag==False)]
        print(classification_report_top1_df[classification_report_top1_df.label==test_label])
        print('测试事件类型={}，测试数据数量={}'.format(test_label, test_data.shape))

        for index in range(0,len(test_data)):
            row = test_data.iloc[index]
            id = row['id']
            label = row['event_label']
            title = row['title']
            content = row['text']
            # print('{}\nindex={}\nid={}\nlabel = {}\ntitle={} \ncontent={}'.format('*' * 100, index,id, label,title, content))
            prediction = event_classifier_request(url,title,content)
            print('{}\nindex={}\nid={}\nlabel = {}\nprediction={}\ntitle={} \ncontent={}'.format('*'*100,index,id,label,prediction,title, content))


def request_post_test_data(url,test_data):
    test_data_ls = []
    for index in range(0, len(test_data)):
        row = test_data.iloc[index]
        id = row['id']
        label = row['label']
        title = row['title']
        content = row['text']
        # print('{}\nindex={}\nid={}\nlabel = {}\ntitle={} \ncontent={}'.format('*' * 100, index,id, label,title, content))
        prediction = event_classifier_request(url, title, content)
        preds = [pred[0] for pred in prediction]
        tag =  True if label in preds else False
        print(
            '{}\nindex={}\nid={}\nlabel = {}\nprediction={}\ntitle={} \ncontent={}'.format('*' * 100, index, id, label,
                                                                                          prediction, title, content))
        row['prediction'] = preds
        row['tag'] =tag
        test_data_ls.append(row)
    test_data_tag = pd.DataFrame(test_data_ls)
    print(test_data_tag.shape)
    print(test_data_tag.tag.value_counts())
    return test_data_tag


def get_test_data(model_test_dir, handed_label_data_predict_path):
    correctified_test_data_path = os.path.join(model_test_dir, "event_test_data_20200927.csv")
    if os.path.exists(correctified_test_data_path):
        correctified_test_data = save_dataframe_to_raw_data(path=correctified_test_data_path)
    else:
        # 读取标注的测试数据的预测结果
        handed_label_data = pd.read_excel(handed_label_data_predict_path, index_col=0)
        correct_predict_handed_label_data = handed_label_data[handed_label_data.tag == True]
        correct_predict_handed_label_data.columns = ['id', 'event_label', 'title', 'text', 'source', 'ori_label',
                                                     'predict_result_whole', 'tag']

        # 读取模型模型测试数据的修正数据
        predict_results_top1_data, classification_report_tail_f1_score = read_model_test_data(model_test_dir,
                                                                                              model_test_result_file='predict_results_top_1_00.xlsx')
        correctified_test_data = pd.concat([predict_results_top1_data, correct_predict_handed_label_data],
                                           ignore_index=True)
        print(correctified_test_data.shape)
        correctified_test_data = correctified_test_data.loc[:,
                                 ['id', 'event_label', 'title', 'text', 'source', 'ori_label']]

        save_dataframe_to_raw_data(dataframe=correctified_test_data, path=correctified_test_data_path)
    return correctified_test_data

if __name__ =='__main__':
    ip = '10.20.33.3'
    url = 'http://10.20.33.3:8015/hsnlp/event_stock/classifier'

    # 单条测试数据
    # test_path = os.path.join('event_classifier', 'corpus/hundsun_juyuan_union_v1', 'important_events.test')
    # raw_test_df = save_dataframe_to_raw_data(path=test_path, mode='r',dataset='event')

    # 读取模型的测试结果并进行规则修正
    from conf.config_parser import OUTPUT_DIR
    model_test_dir = os.path.join(OUTPUT_DIR, 'test_result', '1600840748_p_threshold_0.5')
    predict_results_top1_data,classification_report_tail_f1_score,classification_report_top1_summary_df\
        = read_model_test_data(model_test_dir,model_test_result_file="test_data_predict_results_top_1_13.xlsx")
    # # 预测数据
    test_rule_with_data(predict_results_top1_data, classification_report_tail_f1_score)


    # 读取标注的测试数据 并进行预测
    from build_event_rule import read_handed_label_data
    # data_dir = os.path.join('event_classifier', 'corpus', 'hundsun_juyuan_union_v1')
    # handed_label_data_predict_path = os.path.join(model_test_dir, '27类事件数据补充-20200916-predicted.xlsx')
    # test_data = read_handed_label_data(data_dir)
    # request_post_test_data(url, test_data)
    # test_data_tag.to_excel(handed_label_data_predict_path)

    # 统计测试结果
    correctified_test_data_file = "test_data_predict_results_top_1_12.xlsx"
    evaluation_test_data(test_data_dir=model_test_dir,test_data_file=correctified_test_data_file,
                         url=url)

    # # 分析财务预警反馈数据
    # predict_results_top1_data,classification_report_tail_f1_score,classification_report_top1_summary_df\
    #     = read_model_test_data(model_test_dir,model_test_result_file="feedback_data_predict_results_top_1_1.xlsx")
    # # # 预测数据
    # test_rule_with_data(predict_results_top1_data, classification_report_tail_f1_score)
    #
    # # 统计测试财务预警反馈数据
    # test_data_file = "财务预警用事件标注数据_洪焕杰_0828.xlsx"
    # test_data_path = os.path.join(model_test_dir,test_data_file)
    # test_data = pd.read_excel(test_data_path,sheet_name='Sheet2')
    # print("test_data.shape={}\n原始的测试数据column={}".format(test_data.shape,test_data.columns.tolist()))
    # filter_test_data= test_data.loc[:,['raw_news_id', '新三级事件体系-洪焕杰','event_title', 'news_content','分类是否正确']]
    # filter_test_data = filter_test_data[filter_test_data.loc[:,"分类是否正确"]==0]
    # filter_test_data.columns= ['id','event_label','title','text','tag']
    # print("过滤后的事件数量={},value_counts={}".format(filter_test_data.shape,filter_test_data.event_label.value_counts()))
    #
    # # 将 其他 改为 其他事件类型
    # def filter_event_type(labels,event_type_set):
    #     label_ls = labels.split('@')
    #     filter_labels = []
    #     if label_ls == ['其他']:
    #         return "其他事件类型"
    #     else:
    #         for label in label_ls:
    #             if label in event_type_set:
    #                 filter_labels.append(label)
    #         return "@".join(filter_labels)
    #
    # # 过滤属于新事件体系的数据
    # event_type_set = set(event_type2regular_expression_dict.keys())
    # event_label = filter_test_data.event_label.apply(lambda labels:filter_event_type(labels,event_type_set))
    # filter_test_data.event_label = event_label
    # filter_test_data.dropna(subset=['event_label'],inplace=True)
    # default_other_type_data = filter_test_data.loc[filter_test_data.event_label=="其他事件类型"]
    # print("default_other_type.shape={}".format(default_other_type_data.shape))
    # print("过滤后的事件数量={},value_counts={}".format(filter_test_data.shape,filter_test_data.event_label.value_counts()))
    #
    # evaluation_test_data(test_data=filter_test_data,test_data_dir=model_test_dir ,url=url,test_no=11)




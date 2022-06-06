import os
import json
import math
import numpy as np
import pandas as pd


def create_security_info_dict():
    stock_info_dir = os.path.join('corpus', 'security-info')
    stock_info_file_list = os.listdir(stock_info_dir)
    # 合并美国的上市公司的中文名称信息
    file_name = 'US_SecuMain.csv'
    file_path = os.path.join(stock_info_dir, file_name)
    US_SecuMain_df = pd.read_csv(file_path, index_col=0, dtype=str)
    print('{} shape:{} from {}'.format(file_name, US_SecuMain_df.shape,file_path))

    file_name = 'US_CompanyInfo.csv'
    file_path = os.path.join(stock_info_dir, file_name)
    header = ['ID', 'CompanyCode', 'EngName', 'EngNameAbbr', 'ChiName', 'CompanyType', 'PEOAddress', 'PEOCity',
              'PEOState', 'PEOZip', 'PEOStatus', 'PEOTel', 'EstablishmentDate', 'BusinessDcrp', 'BriefIntroText',
              'UpdateTime', 'JSID']
    US_CompanyInfo_df = pd.read_csv(file_path, header=None, dtype=str)
    US_CompanyInfo_df.columns = header
    print('{} shape from {}:{}'.format(file_name, US_CompanyInfo_df.shape,file_path))


    US_CompanyInfo_SecuMain_df = pd.merge(left=US_SecuMain_df, right=US_CompanyInfo_df, how='left', on=['CompanyCode'])
    US_CompanyInfo_SecuMain_df = US_CompanyInfo_SecuMain_df.drop_duplicates()
    print('US_CompanyInfo_SecuMain_df.shape :{}'.format(US_CompanyInfo_SecuMain_df.shape))

    # 合并中国的上市公司的特殊名称：比如： ST东数
    file_name = 'SecuMain.csv'
    file_path = os.path.join(stock_info_dir, file_name)
    SecuMain_df = pd.read_csv(file_path, index_col=0, dtype=str)
    print('{} shape:{} from {}'.format(file_name, SecuMain_df.shape, file_path))

    file_name = 'LC_SecuChange.csv'
    file_path = os.path.join(stock_info_dir, file_name)
    header = ['ID', 'InnerCode', 'InfoPublDate', 'InfoSource', 'SMDeciPbulDate', 'IfPassed',
              'ChangeDate', 'SecurityAbbr', 'ChiSpelling', 'ChangeReason', 'XGRQ', 'JSID', ]
    LC_SecuChange_df = pd.read_csv(file_path, header=None, dtype=str)
    LC_SecuChange_df.columns = header
    print('{} shape:{}'.format(file_name, LC_SecuChange_df.shape,file_path))

    file_name = 'LC_SpecialTrade.csv'
    file_path = os.path.join(stock_info_dir, file_name)
    header = ['ID', 'InnerCode', 'InfoPublDate', 'SecurityAbbr',
              'SpeicialTradeType', 'SpeicialTradeTime', 'SpeicialTradeExplain',
              'SpeicialTradeReason', 'XGRQ', 'JSID', ]
    LC_SpecialTrade_df = pd.read_csv(file_path, header=None, dtype=str)
    LC_SpecialTrade_df.columns = header
    print('{} shape:{} from {}'.format(file_name, LC_SpecialTrade_df.shape, file_path))

    SecuMain_SecuChange_df = pd.merge(left=SecuMain_df, right=LC_SecuChange_df.loc[:, ['InnerCode', 'SecurityAbbr']],
                                      on=['InnerCode'], how='left')
    SecuMain_SecuChange_df_drop = SecuMain_SecuChange_df.drop_duplicates()

    SecuMain_SecuChange_SpecialTrade_df = pd.merge(left=SecuMain_SecuChange_df_drop,
                                                   right=LC_SpecialTrade_df.loc[:, ['InnerCode', 'SecurityAbbr']],
                                                   on=['InnerCode'], how='left')

    SecuMain_SecuChange_SpecialTrade_df_drop = SecuMain_SecuChange_SpecialTrade_df.drop_duplicates()
    print('SecuMain_SecuChange_SpecialTrade_df_drop.shape:{}'.format(SecuMain_SecuChange_SpecialTrade_df_drop.shape))


    stock_info_files_left = ['HK_SecuMain.csv','NQ_SecuMain.csv']
    df_list = []
    for file_name in  stock_info_files_left:
        #     file_name = stock_info_file_list[0]
        file_path = os.path.join(stock_info_dir, file_name)

        df = pd.read_csv(file_path, index_col=0, dtype=str)
        print('{}\n{} shape:{}'.format('*' * 100, file_name, df.shape))
        df_list.append(df)
    df_list.extend([SecuMain_SecuChange_SpecialTrade_df_drop, US_CompanyInfo_SecuMain_df])

    # 生成一个字典：
    # key作为索引（包括所有 中文全称，简称，证券名称），value为 [ SecuCode	ChiName	ChiNameAbbr	EngName	SecuAbbr]
    security_index_dict = {}
    for df in df_list:
        security_index_dict = add_security_index_dict(security_index_dict, df)

    # 建立二级索引
    security_name_hash_dict = {}
    hash_security_info_dict = {}
    for key, value in security_index_dict.items():
        value_hash_str = str(hash(json.dumps(value)))
        security_name_hash_dict.update({key: value_hash_str})
        hash_security_info_dict.update({value_hash_str: value})
    print('len(security_name_hash_dict):{}'.format(len(security_name_hash_dict)))
    print('len(hash_security_info_dict):{}'.format(len(hash_security_info_dict)))

    # 保存二级索引
    output_dir  = os.path.join('corpus','huatai-event-entity-extract','security_info')
    security_name2hash_dict_path = os.path.join(output_dir,'security_name2hash_dict.json')
    with open(security_name2hash_dict_path, mode='w', encoding='utf-8') as f:
        json.dump(security_name_hash_dict, f, ensure_ascii=False, indent=4)

    hash2security_info_dict_path = os.path.join(output_dir, 'hash2security_info_dict.json')
    with open(hash2security_info_dict_path, mode='w', encoding='utf-8') as f:
        json.dump(hash_security_info_dict, f, ensure_ascii=False, indent=4)
    return security_name_hash_dict,hash_security_info_dict


def update_each_security_info(company_name, new_security_info_dict, security_index_dict):
    """
    针对 company_name,从security_index_dict中提取原始信息，使用 security_info_dict 更新信息
    :param company_name:
    :param ori_security_info:
    :param security_info_dict:
    :return:
    """
    # 提取 security 原来的信息
    ori_security_info = security_index_dict.get(company_name, dict())

    # 初始化：
    # 更新security 信息
    # new_security_info = {}
    # 对提取信息的每个key加入原始的信息中，
    # 为防止 security_info_dict.keys 少于 ori_security_info.keys的情况而导致更新后丢失原始信息的数据，因此不建议使用
    # new_security_info 来重新生成数据，而是在原始数据 ori_security_info做更新

    for key in new_security_info_dict.keys():
        security_info_each_key = new_security_info_dict.get(key)
        if security_info_each_key is not None and type(security_info_each_key) is not float:
            # 获取原始 security信息的对应的key---对应的值
            ori_security_info_each_key = ori_security_info.get(key, [])
            new_security_info_each_key = set(ori_security_info_each_key)
            new_security_info_each_key.add(security_info_each_key)
            new_security_info_each_key = list(new_security_info_each_key)
            # 不建议使用new_security_info 来重新生成数据，而是在原始数据 ori_security_info做更新
            ori_security_info.update({key: new_security_info_each_key})
    return ori_security_info


def add_security_index_dict(security_index_dict,df):
    print('{}\n开始更新 security_index_dict of {} with df.shape:{}'.format('*' * 100, len(security_index_dict), df.shape))
    for i in range(len(df)):
        if i % 1000 ==0:
            print('df.shape:{},开始第{}数据的处理'.format(df.shape,i))
        row = df.iloc[i]
        # 希望获取关于 security的 key
        keys = ['SecuCode', 'ChiName', 'ChiNameAbbr', 'EngName', 'EngNameAbbr', 'SecuAbbr', 'FormerName','CompanyCode',
                'SecurityAbbr_x','SecurityAbbr_y']
        # 获取 对应 security_info_dict
        new_security_info_dict = {}
        for key in keys:
            # 每个key对应的值
            value = row.get(key)
            if value is not None :
                if isinstance(value, float) and math.isnan(value):
                    pass
                else:
                    if isinstance(value, str):
                        value = value.strip().lower()
                    if key in ['SecuAbbr', 'SecurityAbbr_x', 'SecurityAbbr_y']:
                        value = ''.join(value.split())
                    # 将 'SecurityAbbr_x','SecurityAbbr_y' 作为和 'SecuAbbr' 同一个key之中
                    if key in ['SecurityAbbr_x','SecurityAbbr_y']:
                        key ='SecuAbbr'
                    new_security_info_dict.update({key: value})

        # 筛选公司名称的信息，作为检索的key
        new_company_names_set = set()
        filter_keys = ['ChiName', 'ChiNameAbbr', 'EngName', 'EngNameAbbr', 'SecuAbbr','FormerName']
        for key in filter_keys:
            new_company_name = new_security_info_dict.get(key)
            if new_company_name is not None and type(new_company_name) is not float:
                new_company_names_set.add(new_company_name)
        #     print('values_set:{}'.format(values_set))

        # 对每个作为 index 的 new_company_names +  其相关的 relate_company_names，对其信息 进行更新
        relate_company_names = []
        for new_company_name in new_company_names_set:
            # 对于 company_name的 ori_security_info 中提高相关的名称 也要更新
            new_company_name_info_dict = security_index_dict.get(new_company_name)
            if new_company_name_info_dict is not None:
                for key in filter_keys:
                    relate_name_each_key_list = new_company_name_info_dict.get(key)
                    if relate_name_each_key_list is not None:
                        relate_company_names.extend(relate_name_each_key_list)
        # 合并 新增的 new_company_names_set公司名称 和 相关的公司名称 relate_company_names，对其都更加 new_security_info_dict进行更新
        update_company_names = new_company_names_set.union(set(relate_company_names))

        # 对于update_company_names中的company_name，我们认为其应该更新的信息是一致的，对其其中一个进行信息更新,也等于对其他company_name进行更新
        ori_security_info = update_each_security_info(list(update_company_names)[0], new_security_info_dict, security_index_dict)
        for company_name in update_company_names:
            # 增加到信息索引中
            # security_index_dict.update({company_name: new_security_info})
            security_index_dict.update({company_name: ori_security_info})


            # if len(company_relate_names) >0:
            #     for relate_name in company_relate_names:
            #         # 对于 company_relate_names 进行信息更新
            #         relate_name_ori_security_info = update_security_info(relate_name, security_info_dict,security_index_dict)
            #         # 增加到信息索引中
            #         # security_index_dict.update({company_name: new_security_info})
            #         security_index_dict.update({relate_name: relate_name_ori_security_info})

    print('len(security_index_dict):{}'.format(len(security_index_dict)))
    return security_index_dict


def get_security_info(security_name,
                      security_name2hash_dict,
                      hash2security_info_dict):
    value_hash = security_name2hash_dict.get(security_name)
    security_info = hash2security_info_dict.get(value_hash)
    # print('value_hash:{},security_info={}'.format(value_hash,security_info))
    return(security_info,value_hash)


def update_secutity_info(ori_security_info_date = '',
                         new_security_info_date = '-20200517'):
    # 读取 二级索引
    output_dir  = os.path.join('corpus','huatai-event-entity-extract','security_info')
    security_name2hash_dict_path = os.path.join(output_dir,
                                                'security_name2hash_dict'+ori_security_info_date +'.json' )
    with open(security_name2hash_dict_path, mode='r', encoding='utf-8') as f:
        SECURITY_NAME2HASH_DICT = json.load(f)

    hash2security_info_dict_path = os.path.join(output_dir,
                                                'hash2security_info_dict'+ori_security_info_date +'.json')
    with open(hash2security_info_dict_path, mode='r', encoding='utf-8') as f:
        HASH2SECURITY_INFO_DICT = json.load(f)

    ori_len_security_name2hash_dict= len(SECURITY_NAME2HASH_DICT)
    ori_len_hash2security_info_dict = len(HASH2SECURITY_INFO_DICT)
    print('len(SECURITY_NAME2HASH_DICT)={},len(HASH2SECURITY_INFO_DICT)={}'
          .format(ori_len_security_name2hash_dict,ori_len_hash2security_info_dict))

    # 需要更新的数据集：测试数据
    test_dir = os.path.join('corpus', 'huatai-event-entity-extract', 'event_entity_extract_data_0512_04')
    test_path = os.path.join(test_dir,'test.xlsx')
    raw_test_df = pd.read_excel(test_path,index_col=0,dtype=str)

    for index in raw_test_df.index:
        security_name = raw_test_df.loc[index,'entity'].lower()
        security_info,hash_value = get_security_info(security_name,SECURITY_NAME2HASH_DICT,HASH2SECURITY_INFO_DICT)
        print('{}\nindex:{},security_name:{},hash_value:{},\nsecurity_info:{}'.format('*'*100,index,security_name,hash_value,
                                                                                      security_info))
        # hash_value为空且 security_name 长度必须小于20（避免异常的数据进入字典）
        if hash_value is None and len(security_name)<20:
            new_hash_value = str(hash(security_name))
            print('Before:len(SECURITY_NAME2HASH_DICT)={},len(HASH2SECURITY_INFO_DICT)={}'
                  .format(len(SECURITY_NAME2HASH_DICT), len(HASH2SECURITY_INFO_DICT)))
            # 更新字典信息
            SECURITY_NAME2HASH_DICT.update({security_name:new_hash_value})
            HASH2SECURITY_INFO_DICT.update({new_hash_value:{'ChiName':security_name}})
            print('Update : len(SECURITY_NAME2HASH_DICT)={},len(HASH2SECURITY_INFO_DICT)={}'
                  .format(len(SECURITY_NAME2HASH_DICT),len(HASH2SECURITY_INFO_DICT)))

            # 验证更新后的字典对应的key值是否存在
            new_info,new_hash = get_security_info(security_name,SECURITY_NAME2HASH_DICT,HASH2SECURITY_INFO_DICT)
            assert new_info is not None
            assert new_hash is not None

    new_len_security_name2hash_dict = len(SECURITY_NAME2HASH_DICT)
    new_len_hash2security_info_dict = len(HASH2SECURITY_INFO_DICT)
    print('ori vs new len_security_name2hash_dict:{} vs. {}\nori vs. new len_hash2security_info_dict:{} vs. {}'
          .format(ori_len_security_name2hash_dict,new_len_security_name2hash_dict,
                  ori_len_hash2security_info_dict,new_len_hash2security_info_dict))
    # 读取 二级索引
    output_dir  = os.path.join('corpus','huatai-event-entity-extract','security_info')
    security_name2hash_dict_path = os.path.join(output_dir, 'security_name2hash_dict'+new_security_info_date +'.json')
    with open(security_name2hash_dict_path, mode='w', encoding='utf-8') as f:
        json.dump(SECURITY_NAME2HASH_DICT,f, ensure_ascii=False, indent=4)

    hash2security_info_dict_path = os.path.join(output_dir, 'hash2security_info_dict'+new_security_info_date +'.json')
    with open(hash2security_info_dict_path, mode='w', encoding='utf-8') as f:
       json.dump(HASH2SECURITY_INFO_DICT,f, ensure_ascii=False, indent=4)
    return SECURITY_NAME2HASH_DICT,HASH2SECURITY_INFO_DICT

def update_security_name2hash_dict(ori_dict_path, new_dict_path, new_entity_list):
    os.makedirs(os.path.dirname(new_entity_dict_path),exist_ok=True)

    with open(ori_dict_path, mode='r', encoding='utf-8') as f:
        ORI_SECURITY_NAME2HASH_DICT = json.load(f)

    ori_security_name_list=[entity.strip() for entity in list(ORI_SECURITY_NAME2HASH_DICT.keys())]
    ori_security_name_list.extend(new_entity_list)
    updated_entity_set = set(ori_security_name_list)
    sorted_updated_entity_set = sorted(updated_entity_set)

    updated_entity_dict = {security: index for index, security in enumerate(sorted_updated_entity_set)}
    with open(new_dict_path, mode='w', encoding='utf-8') as f:
        json.dump(updated_entity_dict, f, ensure_ascii=False, indent=4)
    print('原始的 security_name_dict 长度为= {} ，更新后长度为={}，保存在 {}'.format(len(ORI_SECURITY_NAME2HASH_DICT),
                                                                    len(updated_entity_dict), new_dict_path))


def update_entity_names_dict_by_entity_extract_data():
    """
    @author:wangfc27441
    @desc:
    @version：
    @time:2020/8/17 13:55

    """
    # 读取 entity 的数据获取新的entity 名称
    new_entity_list = []
    entity_extract_data_path= os.path.join('corpus','entity_extract_data','entity_extract_data_20200811.json')
    with open(entity_extract_data_path,mode = 'r',encoding='utf-8') as f:
        entity_extract_data_json = json.load(f)
        entity_extract_data = entity_extract_data_json['data']
    for example in entity_extract_data:
        paragraphs = example['paragraphs']
        for paragraph in paragraphs:
            qas = paragraph['qas']
            for qa in qas:
                answers= qa['answers']
                for answer_dict in answers:
                    answer = answer_dict['text']
                    new_entity_list.append(answer.lower())
    print('len(new_entity_list)={}'.format(len(new_entity_list)))

    ori_security_info_date= '-20200517'
    output_dir  = os.path.join('corpus','huatai-event-entity-extract','security_info')
    ori_security_name2hash_dict_path = os.path.join(output_dir,
                                                'security_name2hash_dict'+ori_security_info_date +'.json' )
    update_date = '20200817'
    new_entity_dict_path = os.path.join('corpus','security_info','entity_names_'+update_date +'.json' )

    update_security_name2hash_dict(ori_dict_path= ori_security_name2hash_dict_path,
                                   new_dict_path= new_entity_dict_path,
                                   new_entity_list=new_entity_list )




if __name__ == '__main__':

    # 读取 二级索引
    # ori_security_info_date= '-20200517'
    # output_dir  = os.path.join('corpus','huatai-event-entity-extract','security_info')
    # security_name2hash_dict_path = os.path.join(output_dir,
    #                                             'security_name2hash_dict'+ori_security_info_date +'.json' )
    # with open(security_name2hash_dict_path, mode='r', encoding='utf-8') as f:
    #     SECURITY_NAME2HASH_DICT = json.load(f)

    # hash2security_info_dict_path = os.path.join(output_dir,
    #                                             'hash2security_info_dict'+ori_security_info_date +'.json')
    # with open(hash2security_info_dict_path, mode='r', encoding='utf-8') as f:
    #     HASH2SECURITY_INFO_DICT = json.load(f)

    # ori_len_security_name2hash_dict= len(SECURITY_NAME2HASH_DICT)
    # ori_len_hash2security_info_dict = len(HASH2SECURITY_INFO_DICT)
    # print('len(SECURITY_NAME2HASH_DICT)={},len(HASH2SECURITY_INFO_DICT)={}'
    #       .format(ori_len_security_name2hash_dict,ori_len_hash2security_info_dict))
    # print()

    new_entity_list=[]
    abb2full_entity_dict_path = os.path.join('corpus','entity_info','abbr2full.json')
    company_dict_path = os.path.join('corpus','entity_info','company_default_label.json')
    with open(abb2full_entity_dict_path,mode='r',encoding='utf-8') as f:
        abb2full_entity_dict = json.load(f)
        print('len(abb2full_entity_dict)={}'.format(len(abb2full_entity_dict)))

    for length,value in abb2full_entity_dict.items():
        for abb_entity,full_entity in value.items():
            new_entity_list.extend([abb_entity.lower().strip(),full_entity.lower().strip()])
    new_entity_list = list(set(new_entity_list))
    print('合并 abb2full_entity_dict 数据后，len(new_entity_list)={}'.format(len(new_entity_list)))

    with open(company_dict_path,mode='r',encoding='utf-8') as f:
        company_dict = json.load(f)
        print('len(company_dict)'.format(len(company_dict)))

    for key,value in company_dict.items():
        for companies,index in value.items():
            companies_list = companies.split('###')
            new_entity_list.extend([company.lower().strip() for company in companies_list])
    new_entity_list = list(set(new_entity_list))
    print('合并company_dict后，len(new_entity_list)={}'.format(len(new_entity_list)))

    ori_dict_date= '20200817'
    ori_entity_dict_path = os.path.join('corpus','entity_extract_data','entity_names_dict_'+ori_dict_date +'.json' )
    update_dict_date = '20200818'
    new_entity_dict_path = os.path.join('corpus','entity_extract_data','entity_names_dict_'+update_dict_date +'.json' )

    update_security_name2hash_dict(ori_dict_path= ori_entity_dict_path,
                                   new_dict_path= new_entity_dict_path,
                                   new_entity_list=new_entity_list )
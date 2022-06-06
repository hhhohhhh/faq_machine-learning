#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/24 10:48 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/24 10:48   wangfc      1.0         None

在使用中如果发现需要使用多个文件中的fixture，则可以将fixture写入 conftest.py中。
使用时不需要导入要在测试中使用的fixture，它会自动被 pytest 发现。

"""

import pytest



@pytest.fixture()
def regex_pattern_tuple():
    return [("我?想?开通创业[板|版]","想?开通{1,2}创业板?")]



# @pytest.fixture()装饰器用于声明函数是一个fixture。如果测试函数的参数列表中包含fixture名，
# 那么pytest会检测到，并在测试函数运行之前执行fixture。
@pytest.fixture()
def regex_test_data():
    account_operation = "开户+开户操作"
    account_operation_data = ["我想开通创业板。", "如何开通创业板权限？", "要重新开通创业板。", "开通创业板。", "我要开通。创业板问问怎么开。",
                              "港股。港股通港。", "创业板开通。", "创业板开通创业板。", "对呀。开通开通。开通创业板。", "我想申请买创业板的股票怎么处理？",
                              " 放了创业板的。创业板可以开。", "这个。我。我想开通创业板。", " 嗯，开通。嗯，开通创业板。", "对。怎么开通创业板？",
                              "怎么样开通创业板？", "创业板怎么开通？", "我要开通创业板。", "哦，开通创业板。", "怎样开通创业板？", "创业板开开通。",
                              "怎么开通北交所的权限？", "怎么开通北交所权限？", "北交所开通审核。", "嗯，科创板权限开通。", "开户进度查询。"]
    # 银证转账操作 没有被识别
    bank_operation = "银行转账+银行转账操作"
    bank_operation_data = ["我要把资金转到银行卡。", "银行转账。银行转账。", "我怎么把资金转到我绑定的银行卡？", "呃，我想把钱转到银行。",
                           "资金转出。资金转出。", "证券账户资金转到银行。", "如何？把钱转出来。", " 怎么才能把钱转出来？", "咨询资金转出。"]

    # 查询存管银行 被识别为 变更存管银行
    check_bank = "银证转账+查询存管银行"
    check_bank_data = [ "第三方存管。", "没有我这个银行的银行的卡，绑定的银行卡。", " 嗯，我的那个。呃，银行绑定的银行卡。",
    "第三方存管打不开。", " 三方存管业务。三方存管业务。",   " 银行。第三方银行卡的。"]


    # # 转账失败 被识别为 银证转账操作
    transition_fail= "银证转账+转账失败"
    transition_fail_data=[ "资资资金转入出出差错。","我那个资金转出怎么转不出来？"]
    data_dict = {account_operation: account_operation_data,
                 bank_operation: bank_operation_data,
                 check_bank: check_bank_data,
                 transition_fail:transition_fail_data
                 }
    return data_dict

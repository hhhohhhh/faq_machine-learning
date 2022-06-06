#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/11/30 11:27 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/11/30 11:27   wangfc      1.0         None
"""
import os
from typing import List, Text

import tqdm
import numpy as np
from utils.io import read_json_file, dump_obj_as_json_to_file




def get_rasa_nlu_parse_results(questions:List[Text],path,url="http://10.20.33.3:9001/nlu/parse"):
    from apps.rasa_apps.rasa_application import RasaApplication
    if os.path.exists(path):
        prelabel_results = read_json_file(json_path=path)
    else:
        prelabel_results = []
        for index in tqdm.trange(questions.__len__()):
            question = questions[index]
            # count = standard_to_extend_question_num_count.loc[standard_question]
            # 返回一个 字典：  {standard_question: , count:, predict_intent:, predict_entity_type: , predict_entity_value:,}
            result = RasaApplication.get_nlu_parse_result(
                index=index,
                question=question,
                url=url
            )
            result.update({"question": question})

            prelabel_results.append(result)
        dump_obj_as_json_to_file(json_path=path, json_object=prelabel_results)
    return prelabel_results

    # 对 scenario 为空的，默认为 haitong_ivr


def set_scenario(scenario):
    default_scenario = "haitong_ivr"
    if scenario is None:
        scenario = default_scenario
    elif isinstance(scenario, str) and scenario.strip() == "":
        scenario = default_scenario
    if isinstance(scenario, float) and np.isnan(scenario):
        scenario = default_scenario
    return scenario


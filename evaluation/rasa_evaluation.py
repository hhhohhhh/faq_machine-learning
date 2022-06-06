#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/10/8 11:51 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/10/8 11:51   wangfc      1.0         None
"""
import os
from copy import copy, deepcopy

from utils.io import dataframe_to_file
import pandas as pd
from typing import Text, Optional, Dict, Tuple, List, Set

from rasa.utils.io import pickle_dump,pickle_load

from rasa.nlu.constants import RESPONSE_SELECTOR_PROPERTY_NAME, RESPONSE_SELECTOR_DEFAULT_INTENT, \
    RESPONSE_SELECTOR_PREDICTION_KEY, TOKENS_NAMES
from tqdm import tqdm


import rasa
from rasa.shared.nlu.constants import INTENT_RESPONSE_KEY, INTENT, TEXT, INTENT_NAME_KEY, PREDICTED_CONFIDENCE_KEY, \
    ENTITIES, OUTPUT_INTENT, NO_ENTITY_TAG

from rasa.shared.nlu.training_data.training_data import TrainingData

from rasa import telemetry

from rasa.shared.importers.importer import TrainingDataImporter
from rasa.nlu.model import Interpreter
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.test import run_evaluation, remove_pretrained_extractors, get_eval_data, evaluate_intents, \
    evaluate_response_selections, get_entity_extractors, evaluate_entities, IntentEvaluationResult, \
    ResponseSelectionEvaluationResult, EntityEvaluationResult, is_intent_classifier_present, \
    is_response_selector_present, get_available_response_selector_types, is_entity_extractor_present, \
    _targets_predictions_from, _calculate_report, _dump_report, write_intent_successes, \
    _get_intent_errors, _write_errors, plot_attribute_confidences, align_all_entity_predictions, merge_labels, \
    substitute_labels, NO_ENTITY, write_successful_entity_predictions, collect_incorrect_entity_predictions, \
    EXTRACTORS_WITH_CONFIDENCES, merge_confidences, plot_entity_confidences

import rasa.utils.plotting as plot_utils

import logging
logger = logging.getLogger(__name__)


ATTRIBUTE_MAPPING = {
    "开户操作/开户条件/开户材料/查询开户渠道/开户所需时间/开户费用": "开户操作",
    "销户操作/销户费用/销户材料":"销户操作",
    "查询新股中签/新股中签缴款":"查询新股中签",
    "银证转账操作/查询转账额度":"银证转账操作"
}

async def run_rasa_nlu_evaluation(
        data_path: Text,
        model_path: Text,
        output_directory: Optional[Text] = None,
        successes: bool = False,
        errors: bool = False,
        component_builder: Optional[ComponentBuilder] = None,
        disable_plotting: bool = False,
        report_as_dict: Optional[bool] = None,
        if_mapping_to_first_attribute=True,
) -> Dict:  # pragma: no cover
    """Evaluate intent classification, response selection and entity extraction.

    Args:
        data_path: path to the test data
        model_path: path to the model
        output_directory: path to folder where all output will be stored
        successes: if true successful predictions are written to a file
        errors: if true incorrect predictions are written to a file
        component_builder: component builder
        disable_plotting: if true confusion matrix and histogram will not be rendered
        report_as_dict: `True` if the evaluation report should be returned as `dict`.
            If `False` the report is returned in a human-readable text format. If `None`
            `report_as_dict` is considered as `True` in case an `output_directory` is
            given.

    Returns: dictionary containing evaluation results
    """
    import rasa.shared.nlu.training_data.loading

    test_data_importer = TrainingDataImporter.load_from_dict(
        training_data_paths=[data_path]
    )
    test_data = await test_data_importer.get_nlu_data()

    result: Dict[Text, Optional[Dict]] = {
        # "intent_evaluation": None,
        "first_knowledge": None,
        "attribute_results": None,
        "entity_evaluation": None,
        "response_selection_evaluation": None,
    }
    # get the metadata config from the package data
    interpreter = Interpreter.load(model_path, component_builder)

    first_knowledge_results_path = os.path.join(output_directory,"first_knowledge_results.pkl")
    if not os.path.exists(first_knowledge_results_path):
        interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)
        if output_directory:
            rasa.shared.utils.io.create_directory(output_directory)

        (first_knowledge_results, attribute_results,  response_selection_results, entity_results) = \
            get_rasa_eval_data(interpreter, test_data
        )

        for data,filename in zip((first_knowledge_results,attribute_results,  response_selection_results, entity_results) ,
                                 ["first_knowledge_results.pkl", "attribute_results.pkl","response_selection_results.pkl",
                                  "entity_results.pkl"]):
            if data:
                path = os.path.join(output_directory,filename)
                pickle_dump(filename=path, obj=data)
    else:
        path = os.path.join(output_directory, "first_knowledge_results.pkl")
        if os.path.exists(path):
            first_knowledge_results = pickle_load(filename=path)

        path = os.path.join(output_directory, "attribute_results.pkl")
        if os.path.exists(path):
            attribute_results = pickle_load(filename=path)

        path = os.path.join(output_directory, "response_selection_results.pkl")
        if os.path.exists(path):
            response_selection_results = pickle_load(filename=path)
        else:
            response_selection_results =None

        path = os.path.join(output_directory, "entity_results.pkl")
        if os.path.exists(path):
            entity_results = pickle_load(filename=path)

        # 获取详细结果数据
        attribute_df = pd.DataFrame(attribute_results)

    if attribute_results and if_mapping_to_first_attribute:

        new_attribute_results = []
        for attribute_result in attribute_results:
            new_attribute_result = deepcopy(attribute_result)
            predict_attribute = attribute_result.intent_prediction
            if predict_attribute!="" and predict_attribute is not None:
                for key, value in ATTRIBUTE_MAPPING.items():
                    if predict_attribute in key:
                        new_predict_attribute = value
                        new_attribute_result._replace(intent_prediction=new_predict_attribute)
                        break
            new_attribute_results.append(new_attribute_result)
        assert attribute_results.__len__() ==new_attribute_results.__len__()


    if first_knowledge_results:
        logger.info("First_knowledge evaluation results:")
        result["first_knowledge"] = evaluate_rasa_intents(
            "first_knowledge",
            first_knowledge_results,
            output_directory,
            successes,
            errors,
            disable_plotting,
            report_as_dict=report_as_dict,
        )

    if attribute_results:
        logger.info("Attribute_results evaluation results:")
        result["attribute_results"] = evaluate_rasa_intents(
            "attribute",
            attribute_results,
            output_directory,
            successes,
            errors,
            disable_plotting,
            report_as_dict=report_as_dict,
            first_knowledge_results= first_knowledge_results,  # 按照意图进行区分
        )


    if response_selection_results:
        logger.info("Response selection evaluation results:")
        result["response_selection_evaluation"] = evaluate_response_selections(
            response_selection_results,
            output_directory,
            successes,
            errors,
            disable_plotting,
            report_as_dict=report_as_dict,
        )

    if any(entity_results):
        logger.info("Entity evaluation results:")
        extractors = get_entity_extractors(interpreter)
        result["entity_evaluation"] = evaluate_rasa_entities(
            entity_results,
            extractors,
            output_directory,
            successes,
            errors,
            disable_plotting,
            report_as_dict=report_as_dict,
            first_knowledge_results=first_knowledge_results,  # 按照意图进行区分
        )

    telemetry.track_nlu_model_test(test_data)

    return result


def get_rasa_eval_data(
    interpreter: Interpreter, test_data: TrainingData, if_mapping_to_first_attribute=True
) -> Tuple[
    List[IntentEvaluationResult],List[IntentEvaluationResult],
    List[ResponseSelectionEvaluationResult],
    List[EntityEvaluationResult]
]:
    """Runs the model for the test set and extracts targets and predictions.

    Returns intent results (intent targets and predictions, the original
    messages and the confidences of the predictions), response results (
    response targets and predictions) as well as entity results
    (entity_targets, entity_predictions, and tokens).

    Args:
        interpreter: the interpreter
        test_data: test data

    Returns: intent, response, and entity evaluation results
    """
    logger.info("Running model for predictions:")

    # intent_results, entity_results, response_selection_results = [], [], []
    first_knowledge_results,attribute_results, entity_results, response_selection_results =[], [], [], []

    response_labels = [
        e.get(INTENT_RESPONSE_KEY)
        for e in test_data.intent_examples
        if e.get(INTENT_RESPONSE_KEY) is not None
    ]
    intent_labels = [e.get(INTENT) for e in test_data.intent_examples]
    should_eval_intents = (
        is_intent_classifier_present(interpreter) and len(set(intent_labels)) >= 2
    )
    should_eval_response_selection = (
        is_response_selector_present(interpreter) and len(set(response_labels)) >= 2
    )
    available_response_selector_types = get_available_response_selector_types(
        interpreter
    )

    should_eval_entities = is_entity_extractor_present(interpreter)

    for example in tqdm(test_data.nlu_examples):
        result = interpreter.parse(example.get(TEXT), only_output_properties=False)

        if should_eval_intents:
            if rasa.nlu.classifiers.fallback_classifier.is_fallback_classifier_prediction(
                result
            ):
                # Revert fallback prediction to not shadow the wrongly predicted intent
                # during the test phase.
                result = rasa.nlu.classifiers.fallback_classifier.undo_fallback_prediction(
                    result
                )
            # 获取预测结果
            intent_prediction = result.get(INTENT, {}) or {}
            output_intent_prediction = result.get(OUTPUT_INTENT, {})

            # 获取 意图的结果
            true_intent = example.get(INTENT,"")
            true_intent_split = true_intent.split("_")
            if true_intent_split.__len__() ==2:
                true_first_knowledge,true_attribute = true_intent_split
            else:
                true_first_knowledge, true_attribute = true_intent_split[0],""

            first_knowledge_results.append(
                IntentEvaluationResult(
                    true_first_knowledge,
                    output_intent_prediction.get(INTENT_NAME_KEY),
                    result.get(TEXT, {}),
                    output_intent_prediction.get("confidence"),
                )
            )
            # 获取 属性的结果
            predict_intent = intent_prediction.get(INTENT_NAME_KEY)
            predict_intent_split = predict_intent.split("_")

            if predict_intent_split.__len__()==2:
                predict_attribute =  predict_intent_split[-1]
            else:
                predict_attribute = ''

            if if_mapping_to_first_attribute:
                for key,value in ATTRIBUTE_MAPPING.items():
                    if predict_attribute in key:
                        predict_attribute = value
                        break


            attribute_results.append(
                IntentEvaluationResult(
                    true_attribute,
                    predict_attribute,
                    result.get(TEXT, {}),
                    intent_prediction.get("confidence"),
                )
            )

        if should_eval_response_selection:

            # including all examples here. Empty response examples are filtered at the
            # time of metric calculation
            intent_target = example.get(INTENT, "")
            selector_properties = result.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})

            if intent_target in available_response_selector_types:
                response_prediction_key = intent_target
            else:
                response_prediction_key = RESPONSE_SELECTOR_DEFAULT_INTENT

            response_prediction = selector_properties.get(
                response_prediction_key, {}
            ).get(RESPONSE_SELECTOR_PREDICTION_KEY, {})

            intent_response_key_target = example.get(INTENT_RESPONSE_KEY, "")

            response_selection_results.append(
                ResponseSelectionEvaluationResult(
                    intent_response_key_target,
                    response_prediction.get(INTENT_RESPONSE_KEY),
                    result.get(TEXT, {}),
                    response_prediction.get(PREDICTED_CONFIDENCE_KEY),
                )
            )

        if should_eval_entities:
            entity_results.append(
                EntityEvaluationResult(
                    example.get(ENTITIES, []),
                    result.get(ENTITIES, []),
                    result.get(TOKENS_NAMES[TEXT], []),
                    result.get(TEXT, ""),
                )
            )

    return first_knowledge_results,attribute_results,  response_selection_results, entity_results


def evaluate_rasa_intents(
    intent_category:Text,
    intent_results: List[IntentEvaluationResult],
    output_directory: Optional[Text],
    successes: bool,
    errors: bool,
    disable_plotting: bool,
    report_as_dict: Optional[bool] = None,
    first_knowledge_results: List[IntentEvaluationResult]=None
    ) -> Dict:  # pragma: no cover
    """Creates summary statistics for intents.

    Only considers those examples with a set intent. Others are filtered out.
    Returns a dictionary of containing the evaluation result.

    Args:
        intent_results: intent evaluation results
        output_directory: directory to store files to
        successes: if True correct predictions are written to disk
        errors: if True incorrect predictions are written to disk
        disable_plotting: if True no plots are created
        report_as_dict: `True` if the evaluation report should be returned as `dict`.
            If `False` the report is returned in a human-readable text format. If `None`
            `report_as_dict` is considered as `True` in case an `output_directory` is
            given.

    Returns: dictionary with evaluation results
    """
    # remove empty intent targets
    num_examples = len(intent_results)
    # 增加返回 indexes
    intent_results,filtered_indexes = remove_empty_intent_examples(intent_results)

    logger.info(
        f"{intent_category} Evaluation: Only considering those {len(intent_results)} examples "
        f"that have a defined intent out of {num_examples} examples."
    )

    target_intents, predicted_intents = _targets_predictions_from(
        intent_results, "intent_target", "intent_prediction"
    )

    report, precision, f1, accuracy, confusion_matrix, labels = _calculate_report(
        output_directory, target_intents, predicted_intents, report_as_dict,
    )
    if output_directory:
        _dump_report(output_directory, f"{intent_category}_report.json", report)

        report_df = pd.DataFrame(report).T
        path = os.path.join(output_directory,f'{intent_category}_report.xlsx')
        dataframe_to_file(path=path,data=report_df,sheet_name=intent_category)

    if first_knowledge_results:
        # 按照 意图进行分类统计
        first_knowledge_df = pd.DataFrame(first_knowledge_results)
        filter_first_knowledge_df= first_knowledge_df.loc[filtered_indexes].copy()
        filter_first_knowledge_df.reset_index(inplace=True)
        first_knowledge_grouped = filter_first_knowledge_df.groupby("intent_target")

        for first_knowledge, indexes in first_knowledge_grouped.groups.items():
            target_intents_in_group = [target_intents[index] for index in indexes]
            predicted_intents_in_group = [predicted_intents[index] for index in indexes]
            report, precision, f1, accuracy, confusion_matrix, labels = _calculate_report(
                output_directory, target_intents_in_group, predicted_intents_in_group, report_as_dict,
            )
            # 保存report
            report_df = pd.DataFrame(report).T
            dataframe_to_file(path = path,mode='a',data=report_df,
                              sheet_name=f"{first_knowledge}_{intent_category}")


    if successes and output_directory:
        successes_filename = os.path.join(output_directory, f"{intent_category}_successes.json")
        # save classified samples to file for debugging
        write_intent_successes(intent_results, successes_filename)

    intent_errors = _get_intent_errors(intent_results)
    if errors and output_directory:
        errors_filename = os.path.join(output_directory, f"{intent_category}_errors.json")
        _write_errors(intent_errors, errors_filename, f"{intent_category}")

    if not disable_plotting:
        confusion_matrix_filename = f"{intent_category}_confusion_matrix.png"
        if output_directory:
            confusion_matrix_filename = os.path.join(
                output_directory, confusion_matrix_filename
            )
        plot_utils.plot_confusion_matrix(
            confusion_matrix,
            classes=labels,
            title=f"{intent_category} Confusion matrix",
            output_file=confusion_matrix_filename,
        )

        histogram_filename = f"{intent_category}_histogram.png"
        if output_directory:
            histogram_filename = os.path.join(output_directory, histogram_filename)
        plot_attribute_confidences(
            intent_results,
            histogram_filename,
            "intent_target",
            "intent_prediction",
            title=f"{intent_category} Prediction Confidence Distribution",
        )

    predictions = [
        {
            "text": res.message,
            "intent": res.intent_target,
            "predicted": res.intent_prediction,
            "confidence": res.confidence,
        }
        for res in intent_results
    ]

    return {
        "predictions": predictions,
        "report": report,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
        "errors": intent_errors,
    }


def evaluate_rasa_entities(
    entity_results: List[EntityEvaluationResult],
    extractors: Set[Text],

    output_directory: Optional[Text],
    successes: bool,
    errors: bool,
    disable_plotting: bool,
    report_as_dict: Optional[bool] = None,
    first_knowledge_results: List[IntentEvaluationResult]=None,
) -> Dict:  # pragma: no cover
    """Creates summary statistics for each entity extractor.

    Logs precision, recall, and F1 per entity type for each extractor.

    Args:
        entity_results: entity evaluation results
        extractors: entity extractors to consider
        output_directory: directory to store files to
        successes: if True correct predictions are written to disk
        errors: if True incorrect predictions are written to disk
        disable_plotting: if True no plots are created
        report_as_dict: `True` if the evaluation report should be returned as `dict`.
            If `False` the report is returned in a human-readable text format. If `None`
            `report_as_dict` is considered as `True` in case an `output_directory` is
            given.

    Returns: dictionary with evaluation results
    """
    aligned_predictions = align_all_entity_predictions(entity_results, extractors)
    merged_targets = merge_labels(aligned_predictions)
    merged_targets = substitute_labels(merged_targets, NO_ENTITY_TAG, NO_ENTITY)

    result = {}

    for extractor in extractors:
        merged_predictions = merge_labels(aligned_predictions, extractor)
        merged_predictions = substitute_labels(
            merged_predictions, NO_ENTITY_TAG, NO_ENTITY
        )

        logger.info(f"Evaluation for entity extractor: {extractor} ")

        report, precision, f1, accuracy, confusion_matrix, labels = _calculate_report(
            output_directory,
            merged_targets,
            merged_predictions,
            report_as_dict,
            exclude_label=NO_ENTITY,
        )
        if output_directory:
            _dump_report(output_directory, f"{extractor}_report.json", report)
            report_df = pd.DataFrame(report).T
            path = os.path.join(output_directory,f"{extractor}_report.xlsx")
            dataframe_to_file(path = path,data=report_df)

        if first_knowledge_results:
            # 按照 意图进行分类统计
            first_knowledge_df = pd.DataFrame(first_knowledge_results)
            first_knowledge_grouped = first_knowledge_df.groupby("intent_target")
            for first_knowledge, indexes in first_knowledge_grouped.groups.items():
                aligned_predictions_in_group = [aligned_predictions[index] for index in indexes]
                merged_targets = merge_labels(aligned_predictions_in_group)
                merged_targets = substitute_labels(merged_targets, NO_ENTITY_TAG, NO_ENTITY)

                merged_predictions = merge_labels(aligned_predictions_in_group, extractor)
                merged_predictions = substitute_labels(
                    merged_predictions, NO_ENTITY_TAG, NO_ENTITY
                )

                report, precision, f1, accuracy, confusion_matrix, labels = _calculate_report(
                    output_directory,
                    merged_targets,
                    merged_predictions,
                    report_as_dict,
                    exclude_label=NO_ENTITY,
                )
                # 保存report
                report_df = pd.DataFrame(report).T
                dataframe_to_file(path=path, mode='a', data=report_df,
                                  sheet_name=f"{first_knowledge}_{extractor}")


        if successes:
            successes_filename = f"{extractor}_successes.json"
            if output_directory:
                successes_filename = os.path.join(output_directory, successes_filename)
            # save classified samples to file for debugging
            write_successful_entity_predictions(
                entity_results, merged_targets, merged_predictions, successes_filename
            )

        entity_errors = collect_incorrect_entity_predictions(
            entity_results, merged_predictions, merged_targets
        )
        if errors and output_directory:
            errors_filename = os.path.join(output_directory, f"{extractor}_errors.json")

            _write_errors(entity_errors, errors_filename, "entity")

        if not disable_plotting:
            confusion_matrix_filename = f"{extractor}_confusion_matrix.png"
            if output_directory:
                confusion_matrix_filename = os.path.join(
                    output_directory, confusion_matrix_filename
                )
            plot_utils.plot_confusion_matrix(
                confusion_matrix,
                classes=labels,
                title="Entity Confusion matrix",
                output_file=confusion_matrix_filename,
            )

            if extractor in EXTRACTORS_WITH_CONFIDENCES:
                merged_confidences = merge_confidences(aligned_predictions, extractor)
                histogram_filename = f"{extractor}_histogram.png"
                if output_directory:
                    histogram_filename = os.path.join(
                        output_directory, histogram_filename
                    )
                plot_entity_confidences(
                    merged_targets,
                    merged_predictions,
                    merged_confidences,
                    title="Entity Prediction Confidence Distribution",
                    hist_filename=histogram_filename,
                )

        result[extractor] = {
            "report": report,
            "precision": precision,
            "f1_score": f1,
            "accuracy": accuracy,
            "errors": entity_errors,
        }

    return result


def remove_empty_intent_examples(
    intent_results: List[IntentEvaluationResult],
) -> Tuple[List[IntentEvaluationResult],List[int]]:
    """Remove those examples without an intent.

    Args:
        intent_results: intent evaluation results

    Returns: intent evaluation results
    """
    filtered = []
    filtered_indexes = []
    for index,r in enumerate(intent_results):
        # substitute None values with empty string
        # to enable sklearn evaluation
        if r.intent_prediction is None:
            r = r._replace(intent_prediction="")

        if r.intent_target != "" and r.intent_target is not None:
            filtered.append(r)
            filtered_indexes.append(index)
        else:
            logger.info(f"过滤 {r}")

    return filtered,filtered_indexes

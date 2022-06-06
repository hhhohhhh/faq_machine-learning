#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/17 11:11 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/17 11:11   wangfc      1.0         None
"""
from typing import Text, Type, Dict, Any, Optional

import traceback


from utils.common import class_from_module_path
from utils.exceptions import ComponentNotFoundException
from utils.io import raise_deprecation_warning

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.model import Metadata
# from tokenizations.lm_tokenizer import LanguageModelTokenizer
from tokenizations.hf_transformers_zh_tokenizer import HFTransformersTokenizer
from models.featurizers.sparse_featurizer.count_vectors_featurizer import CountVectorsFeaturizer
from models.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from models.classifiers.diet_classifier import DIETClassifier, HierarchicalDIETClassifier
from models.classifiers.intent_mapper import IntentAttributeMapper


component_classes = [
    # utils
    # SpacyNLP,
    # MitieNLP,
    # HFTransformersNLP,


    #
    # tokenizers
    # MitieTokenizer,
    # SpacyTokenizer,
    # WhitespaceTokenizer,
    # ConveRTTokenizer,
    # JiebaTokenizer,
    # LanguageModelTokenizer,
    # 新增 自定义的 tokenzier
    HFTransformersTokenizer,

    # featurizers
    # SpacyFeaturizer,
    # MitieFeaturizer,
    # RegexFeaturizer,
    # LexicalSyntacticFeaturizer,
    CountVectorsFeaturizer,
    # ConveRTFeaturizer,
    LanguageModelFeaturizer,

    # classifiers
    # SklearnIntentClassifier,
    # MitieIntentClassifier,
    # KeywordIntentClassifier,
    DIETClassifier,
    HierarchicalDIETClassifier,
    IntentAttributeMapper
    # IntentToOutputMapper,
    # FallbackClassifier,


    # extractors
    # SpacyEntityExtractor,
    # MitieEntityExtractor,
    # CRFEntityExtractor,
    # DucklingEntityExtractor,
    # EntitySynonymMapper,
    # RegexEntityExtractor,
    # 新增 自定义的 EntitySelector
    # EntitySelector,

    # # selectors
    # ResponseSelector,

]

# Mapping from a components name to its class to allow name based lookup.
registered_components = {c.name: c for c in component_classes}


def get_component_class(component_name: Text) -> Type["Component"]:
    """Resolve component name to a registered components class."""

    if component_name == "DucklingHTTPExtractor":
        raise_deprecation_warning(
            "The component 'DucklingHTTPExtractor' has been renamed to "
            "'DucklingEntityExtractor'. Update your pipeline to use "
            "'DucklingEntityExtractor'.",
            # docs=DOCS_URL_COMPONENTS,
        )
        component_name = "DucklingEntityExtractor"

    if component_name not in registered_components:
        try:
            # return rasa.shared.utils.common.class_from_module_path(component_name)
            return class_from_module_path(component_name)

        except (ImportError, AttributeError) as e:
            # when component_name is a path to a class but that path is invalid or
            # when component_name is a class name and not part of old_style_names

            is_path = "." in component_name

            if is_path:
                module_name, _, class_name = component_name.rpartition(".")
                if isinstance(e, ImportError):
                    exception_message = f"Failed to find module '{module_name}'."
                else:
                    # when component_name is a path to a class but the path does
                    # not contain that class
                    exception_message = (
                        f"The class '{class_name}' could not be "
                        f"found in module '{module_name}'."
                    )
            else:
                exception_message = (
                    f"Cannot find class '{component_name}' in global namespace. "
                    f"Please check that there is no typo in the class "
                    f"name and that you have imported the class into the global "
                    f"namespace."
                )

            raise ComponentNotFoundException(
                f"Failed to load the component "
                f"'{component_name}'. "
                f"{exception_message} Either your "
                f"pipeline configuration contains an error "
                f"or the module you are trying to import "
                f"is broken (e.g. the module is trying "
                f"to import a package that is not "
                f"installed). {traceback.format_exc()}"
            )

    return registered_components[component_name]





def create_component_by_config(
    component_config: Dict[Text, Any], config: "RasaNLUModelConfig"
) -> Optional["Component"]:
    """Resolves a component and calls it's create method.

    Inits it based on a previously persisted model.
    """

    # try to get class name first, else create by name
    component_name = component_config.get("class", component_config["name"])
    component_class = get_component_class(component_name)
    return component_class.create(component_config, config)



def load_component_by_meta(
    component_meta: Dict[Text, Any],
    model_dir: Text,
    metadata: "Metadata",
    cached_component: Optional["Component"],
    **kwargs: Any,
) -> Optional["Component"]:
    """Resolves a component and calls its load method.

    Inits it based on a previously persisted model.
    """

    # try to get class name first, else create by name
    component_name = component_meta.get("class", component_meta["name"])
    component_class = get_component_class(component_name)
    return component_class.load(
        component_meta, model_dir, metadata, cached_component, **kwargs
    )

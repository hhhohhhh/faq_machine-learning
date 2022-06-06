#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 13:48 




@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 13:48   wangfc      1.0         None
"""
import copy
from typing import Optional, Dict, Text, Any, List

from utils.constants import COMPONENT_INDEX
from models.train_utils import override_defaults

from utils.io import json_to_string, raise_warning

# from rasa.nlu.config import RasaNLUModelConfig
class RasaNLUModelConfig:
    """A class that stores NLU model configuration parameters.
    from  rasa/nlu/config/RasaNLUModelConfig
    """

    def __init__(self, configuration_values: Optional[Dict[Text, Any]] = None) -> None:
        """Create a model configuration.

        Args:
            configuration_values: optional dictionary to override defaults.
        """
        if not configuration_values:
            configuration_values = {}

        self.language = "en"
        self.pipeline = []
        self.data = None

        self.override(configuration_values)

        if self.__dict__["pipeline"] is None:
            # replaces None with empty list
            self.__dict__["pipeline"] = []

        for key, value in self.items():
            setattr(self, key, value)

    def __getitem__(self, key: Text) -> Any:
        return self.__dict__[key]

    def get(self, key: Text, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def __setitem__(self, key: Text, value: Any) -> None:
        self.__dict__[key] = value

    def __delitem__(self, key: Text) -> None:
        del self.__dict__[key]

    def __contains__(self, key: Text) -> bool:
        return key in self.__dict__

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getstate__(self) -> Dict[Text, Any]:
        return self.as_dict()

    def __setstate__(self, state: Dict[Text, Any]) -> None:
        self.override(state)

    def items(self) -> List[Any]:
        return list(self.__dict__.items())

    def as_dict(self) -> Dict[Text, Any]:
        return dict(list(self.items()))

    def view(self) -> Text:
        return json_to_string(self.__dict__, indent=4)

    def for_component(
        self, index: int, defaults: Optional[Dict[Text, Any]] = None
    ) -> Dict[Text, Any]:
        return component_config_from_pipeline(index, self.pipeline, defaults)

    @property
    def component_names(self) -> List[Text]:
        if self.pipeline:
            return [c.get("name") for c in self.pipeline]
        else:
            return []

    def set_component_attr(self, index: int, **kwargs: Any) -> None:
        try:
            self.pipeline[index].update(kwargs)
        except IndexError:
            raise_warning(
                f"Tried to set configuration value for component "
                f"number {index} which is not part of the pipeline.",
                # docs=DOCS_URL_PIPELINE,
            )

    def override(self, config: Optional[Dict[Text, Any]] = None) -> None:
        """Overrides default config with given values.

        Args:
            config: New values for the configuration.
        """
        if config:
            self.__dict__.update(config)


class NLUModelConfig(RasaNLUModelConfig):
    """
    继承 RasaNLUModelConfig
    """



def component_config_from_pipeline(
    index: int,
    pipeline: List[Dict[Text, Any]],
    defaults: Optional[Dict[Text, Any]] = None,
) -> Dict[Text, Any]:
    """Gets the configuration of the `index`th component.

    Args:
        index: Index of the component in the pipeline.
        pipeline: Configurations of the components in the pipeline.
        defaults: Default configuration.

    Returns:
        The `index`th component configuration, expanded
        by the given defaults.
    """
    try:
        configuration = copy.deepcopy(pipeline[index])
        configuration[COMPONENT_INDEX] = index
        return override_defaults(defaults, configuration)
    except IndexError:
        raise_warning(
            f"Tried to get configuration value for component "
            f"number {index} which is not part of your pipeline. "
            f"Returning `defaults`.",
            # docs=DOCS_URL_PIPELINE,
        )
        return override_defaults(
            defaults, {COMPONENT_INDEX: index}
        )

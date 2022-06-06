#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/2/16 16:05 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/16 16:05   wangfc      1.0         None
"""
import os
import sys
from typing import Any, Text, NoReturn, Optional
from utils.io import wrap_with_color, bcolors, raise_warning

import logging

logger = logging.getLogger(__name__)


def print_color(*args: Any, color: Text) -> None:
    output = wrap_with_color(*args, color=color)
    try:
        # colorama is used to fix a regression where colors can not be printed on
        # windows. https://github.com/RasaHQ/rasa/issues/7053
        from colorama import AnsiToWin32

        stream = AnsiToWin32(sys.stdout).stream
        print(output, file=stream)
    except ImportError:
        print(output)


def print_success(*args: Any) -> None:
    print_color(*args, color=bcolors.OKGREEN)


def print_info(*args: Any) -> None:
    print_color(*args, color=bcolors.OKBLUE)


def print_warning(*args: Any) -> None:
    print_color(*args, color=bcolors.WARNING)


def print_error(*args: Any) -> None:
    print_color(*args, color=bcolors.FAIL)


def print_error_and_exit(message: Text, exit_code: int = 1) -> NoReturn:
    """Print error message and exit the application."""

    print_error(message)
    sys.exit(exit_code)


def get_validated_path(
        current: Optional[Text],
        parameter: Text,
        default: Optional[Text] = None,
        none_is_valid: bool = False,
) -> Optional[Text]:
    """Check whether a file path or its default value is valid and returns it.

    Args:
        current: The parsed value.
        parameter: The name of the parameter.
        default: The default value of the parameter.
        none_is_valid: `True` if `None` is valid value for the path,
                        else `False``

    Returns:
        The current value if it was valid, else the default value of the
        argument if it is valid, else `None`.
    """
    if current is None or current is not None and not os.path.exists(current):
        if default is not None and os.path.exists(default):
            reason_str = f"'{current}' not found."
            if current is None:
                reason_str = f"Parameter '{parameter}' not set."
            else:
                raise_warning(
                    f"The path '{current}' does not seem to exist. Using the "
                    f"default value '{default}' instead."
                )

            logger.debug(f"{reason_str} Using default location '{default}' instead.")
            current = default
        elif none_is_valid:
            current = None
        else:
            cancel_cause_not_found(current, parameter, default)

    return current


def cancel_cause_not_found(
        current: Optional[Text], parameter: Text, default: Optional[Text]
) -> None:
    """Exits with an error because the given path was not valid.

    Args:
        current: The path given by the user.
        parameter: The name of the parameter.
        default: The default value of the parameter.

    """

    default_clause = ""
    if default:
        default_clause = f"use the default location ('{default}') or "
    print_error(
        "The path '{}' does not exist. Please make sure to {}specify it"
        " with '--{}'.".format(current, default_clause, parameter)
    )
    sys.exit(1)

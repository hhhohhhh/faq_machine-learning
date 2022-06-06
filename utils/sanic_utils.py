#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/14 9:38 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/14 9:38   wangfc      1.0         None
"""
from typing import Text, Optional

from sanic import Sanic
from sanic.views import CompositionView

import logging
logger = logging.getLogger(__name__)


def list_routes(app: Sanic) -> Text:
    """List all the routes of a sanic application.

    Mainly used for debugging."""
    from urllib.parse import unquote

    output = {}

    def find_route(suffix: Text, path: Text) -> Optional[Text]:
        for name, (uri, _) in app.router.routes_names.items():
            if name.split(".")[-1] == suffix and uri == path:
                return name
        return None

    for endpoint, route in app.router.routes_all.items():
        if endpoint[:-1] in app.router.routes_all and endpoint[-1] == "/":
            continue

        options = {}
        for arg in route.parameters:
            options[arg] = f"[{arg}]"

        if not isinstance(route.handler, CompositionView):
            handlers = [(list(route.methods)[0], route.name)]
        else:
            handlers = [
                (method, find_route(v.__name__, endpoint) or v.__name__)
                for method, v in route.handler.handlers.items()
            ]

        for method, name in handlers:
            line = unquote(f"{endpoint:50s} {method:30s} {name}")
            output[name] = line

    url_table = "\n".join(output[url] for url in sorted(output))
    logger.debug(f"Available web server routes: \n{url_table}")

    return output



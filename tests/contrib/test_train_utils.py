# -*- coding: utf-8 -*-
# mypy: ignore-errors

from jaxgp.contrib.train_utils import deep_update_no_new_key as deep_update


def test_deep_update():
    source = {"hello1": 1}
    overrides = {"hello2": 2}
    res = deep_update(source, overrides)
    assert res == {"hello1": 1}

    source = {"hello": "to_override"}
    overrides = {"hello": "over"}
    res = deep_update(source, overrides)
    assert res == {"hello": "over"}

    source = {"hello": {"value": "to_override", "no_change": 1}}
    overrides = {"hello": {"value": "over"}}
    res = deep_update(source, overrides)
    assert res == {"hello": {"value": "over", "no_change": 1}}

    source = {"hello": {"value": "to_override", "no_change": 1}}
    overrides = {"hello": {"value": {}}}
    res = deep_update(source, overrides)
    assert res == {"hello": {"value": {}, "no_change": 1}}

    source = {"hello": {"value": {}, "no_change": 1}}
    overrides = {"hello": {"value": 2}}
    res = deep_update(source, overrides)
    assert res == {"hello": {"value": 2, "no_change": 1}}

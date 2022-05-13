# -*- coding: utf-8 -*-
# mypy: ignore-errors
from jaxgp.utils import deep_update, deep_update_no_new_key


def test_deep_update_no_new_key():
    source = {"hello1": 1}
    overrides = {"hello2": 2}
    res = deep_update_no_new_key(source, overrides)
    assert res == {"hello1": 1}

    source = {"hello": "to_override"}
    overrides = {"hello": "over"}
    res = deep_update_no_new_key(source, overrides)
    assert res == {"hello": "over"}

    source = {"hello": {"value": "to_override", "no_change": 1}}
    overrides = {"hello": {"value": "over"}}
    res = deep_update_no_new_key(source, overrides)
    assert res == {"hello": {"value": "over", "no_change": 1}}

    source = {"hello": {"value": "to_override", "no_change": 1}}
    overrides = {"hello": {"value": {}}}
    res = deep_update_no_new_key(source, overrides)
    assert res == {"hello": {"value": {}, "no_change": 1}}

    source = {"hello": {"value": {}, "no_change": 1}}
    overrides = {"hello": {"value": 2}}
    res = deep_update_no_new_key(source, overrides)
    assert res == {"hello": {"value": 2, "no_change": 1}}


def test_deep_update():
    s = {"1": {"2": [2, 3]}, "3": 3}
    o = {"1": {"2": [1, 3]}}
    res = deep_update(s, o)
    assert res == {"1": {"2": [1, 3]}, "3": 3}
    assert s == {"1": {"2": [2, 3]}, "3": 3}

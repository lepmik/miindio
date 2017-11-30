import pytest
import os
import json
import copy
import xml.etree.ElementTree as ET

xml_path = os.path.join(os.path.dirname(__file__), 'test.xml')
test_path = xml_path.replace('.xml', '') + '_temp'

def test_dictify():
    from tools import dictify, dict_changed
    a = {
        'a': {'b': 'c'},
        'd': ['e', 'f', [1, 2, 'd']],
        'g': 1,
        'h': 'j'
    }
    b = copy.deepcopy(a)
    dictify(a)
    val = {0: 'e', 1: 'f', 2: {0: 1, 1: 2, 2: 'd'}}
    assert dict_changed(a['d'], val) == set()
    assert dict_changed(b, a) == set('d')


def test_listify():
    from tools import listify, dict_changed
    a = {
        'a': {'b': 'c'},
        'd': {0: 'e', 1: 'f', 2: {0: 1, 1: 2, 2: 'd'}},
        'g': 1,
        'h': 'j'
    }
    b = copy.deepcopy(a)
    listify(a)
    assert a['d'] == ['e', 'f', [1, 2, 'd']]
    assert dict_changed(b, a) == set('d')


def test_read_write_dict_equal():
    from tools import convert_xml_dict, dump_xml, to_json, dict_changed
    p = convert_xml_dict(xml_path)
    to_json(p, test_path + '.json')
    p = json.loads(json.dumps(p))
    dump_xml(p, test_path + '.xml')
    q = convert_xml_dict(test_path + '.xml')
    q = json.loads(json.dumps(q))
    assert dict_changed(p, q) == set()


# def test_read_write_string_equal():
#     from tools import convert_xml_dict, dump_xml, to_json, dict_changed
#     p = convert_xml_dict(xml_path)
#     dump_xml(p, test_path + '.xml')
#     q_string = ET.tostring(ET.parse(test_path + '.xml').getroot())
#     p_string = ET.tostring(ET.parse(xml_path).getroot())
#     assert q_string == p_string


def test_set_value_no_content():
    from tools import set_params, convert_xml_dict, dict_changed
    params = {'t_end': .1}
    p = convert_xml_dict(xml_path)
    q = copy.deepcopy(p)
    set_params(q, **params)
    assert dict_changed(p, q) == set()
    set_params(p, t_end=1)
    assert p['Simulation']['SimulationRunParameter']['t_end']['content'] == 1


def test_set_value_no_content_deep():
    from tools import set_params, convert_xml_dict
    params = {
        'Algorithm': {
            2: {'expression': 1.}
        }
    }
    p = convert_xml_dict(xml_path)
    set_params(p, **params)
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 1.
    assert base['type'] == "RateFunctor"


def test_set_no_content_deep_string():
    from tools import set_params, convert_xml_dict
    p = convert_xml_dict(xml_path)
    set_params(p, Algorithm='2/expression/1.')
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 1.
    assert base['type'] == "RateFunctor"


def test_change_params_dict():
    from tools import set_params, convert_xml_dict
    params = {
        'Connection': {
            3: {'content': '-1000 100. 0001'}
        }
    }
    p = convert_xml_dict(xml_path)
    set_params(p, **params)
    base = p['Simulation']['Connections']['Connection'][3]
    assert base['content'] == '-1000 100. 0001'
    assert base['In'] == 'adex E'
    assert base['Out'] == 'adex I'


def test_change_params_dict2():
    from tools import set_params, convert_xml_dict
    params = {
        'Algorithm': {
            2: {'expression': {'content': 1.}}
        }
    }
    p = convert_xml_dict(xml_path)
    set_params(p, **params)
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 1.
    assert base['type'] == "RateFunctor"


def test_change_params_string():
    from tools import set_params, convert_xml_dict
    p = convert_xml_dict(xml_path)
    set_params(p, Algorithm='2/expression/content/1.')
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 1.
    assert base['type'] == "RateFunctor"


def test_set_val_new_type():
    from tools import set_params, convert_xml_dict
    p = convert_xml_dict(xml_path)
    with pytest.raises(TypeError):
        set_params(p, Algorithm='2/expression/1')


def test_set_val_new_structure():
    from tools import set_params, convert_xml_dict
    p = convert_xml_dict(xml_path)
    with pytest.raises(TypeError):
        set_params(p, Algorithm='2/1')


def test_set_attr():
    from tools import set_params, convert_xml_dict
    p = convert_xml_dict(xml_path)
    set_params(p, Node='0/algorithm/yoyo')
    assert p['Simulation']['Nodes']['Node'][0]['algorithm'] == 'yoyo'


def test_set_multiple_values():
    from tools import set_params, convert_xml_dict, dump_xml
    params = {'Algorithm': '2/expression/2000.',
              'Connection': {2: '50 -1 1',
                             3: '200 1 1'}}
    p = convert_xml_dict(xml_path)
    set_params(p, **params)
    dump_xml(p, test_path + '.xml')
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 2000.

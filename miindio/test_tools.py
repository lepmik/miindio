import pytest
import os
import json
import copy
import xml.etree.ElementTree as ET

xmlpath = os.path.join(os.path.dirname(__file__), 'test.xml')
testpath = xmlpath.replace('.xml', '') + '_temp'

def test_dictify():
    from tools import dictify, DictDiffer
    a = {
        'a': {'b': 'c'},
        'd': ['e', 'f', [1, 2, 'd']],
        'g': 1,
        'h': 'j'
    }
    b = copy.deepcopy(a)
    dictify(a)
    val = {0: 'e', 1: 'f', 2: {0: 1, 1: 2, 2: 'd'}}
    assert DictDiffer(a['d'], val).changed() == set()
    assert DictDiffer(b, a).changed() == set('d')


def test_listify():
    from tools import listify, DictDiffer
    a = {
        'a': {'b': 'c'},
        'd': {0: 'e', 1: 'f', 2: {0: 1, 1: 2, 2: 'd'}},
        'g': 1,
        'h': 'j'
    }
    b = copy.deepcopy(a)
    listify(a)
    assert a['d'] == ['e', 'f', [1, 2, 'd']]
    assert DictDiffer(b, a).changed() == set('d')


def test_read_write_dict_equal():
    from tools import convert_xml_dict, dump_xml, to_json, DictDiffer
    p = convert_xml_dict(xmlpath)
    to_json(p, testpath + '.json')
    p = json.loads(json.dumps(p))
    dump_xml(p, testpath + '.xml')
    q = convert_xml_dict(testpath + '.xml')
    q = json.loads(json.dumps(q))
    assert DictDiffer(p, q).changed() == set()


def test_read_write_string_equal():
    from tools import convert_xml_dict, dump_xml, to_json, DictDiffer
    p = convert_xml_dict(xmlpath)
    dump_xml(p, testpath + '.xml')
    q_string = ET.tostring(ET.parse(testpath + '.xml').getroot())
    p_string = ET.tostring(ET.parse(xmlpath).getroot())
    assert q_string == p_string


def test_set_content():
    from tools import set_params, convert_xml_dict, DictDiffer
    params = {
        't_end': .1,
    }
    p = convert_xml_dict(xmlpath)
    q = copy.deepcopy(p)
    set_params(q, **params)
    assert DictDiffer(p, q).changed() == set()
    set_params(p, t_end=1)
    assert p['Simulation']['SimulationRunParameter']['t_end']['content'] == 1


def test_change_list_params1():
    from tools import set_params, convert_xml_dict
    params = {
        'Connection': {
            3: {'content': '-1000 100. 0001'}
        }
    }
    p = convert_xml_dict(xmlpath)
    set_params(p, **params)
    base = p['Simulation']['Connections']['Connection'][3]
    assert base['content'] == '-1000 100. 0001'
    assert base['In'] == 'adex E'
    assert base['Out'] == 'adex I'


def test_change_list_params2():
    from tools import set_params, convert_xml_dict
    params = {
        'Algorithm': {
            2: {'expression': {'content': 1.}}
        }
    }
    p = convert_xml_dict(xmlpath)
    set_params(p, **params)
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 1.
    assert base['type'] == "RateFunctor"


def test_change_list_params_no():
    from tools import set_params, convert_xml_dict
    params = {
        'Algorithm': {
            2: {'expression': 1}
        }
    }
    p = convert_xml_dict(xmlpath)
    with pytest.raises(TypeError):
        set_params(p, **params)

import pytest
import os

xmlpath = os.path.join(os.path.dirname(__file__), 'test.xml')


class DictDiffer(object):
    """
    A dictionary difference calculator
    Originally posted as:
    http://stackoverflow.com/questions/1165352/fast-comparison-between-two-python-dictionary/1165552#1165552

    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values
    """
    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.current_keys, self.past_keys = [
            set(d.keys()) for d in (current_dict, past_dict)
        ]
        self.intersect = self.current_keys.intersection(self.past_keys)

    def added(self):
        return self.current_keys - self.intersect

    def removed(self):
        return self.past_keys - self.intersect

    def changed(self):
        return set(o for o in self.intersect
                   if self.past_dict[o] != self.current_dict[o])

    def unchanged(self):
        return set(o for o in self.intersect
                   if self.past_dict[o] == self.current_dict[o])


def test_read_write():
    from core import convert_xml_dict, dump_xml
    p = convert_xml_dict(xmlpath)
    dump_xml(p, xmlpath)
    q = convert_xml_dict(xmlpath)
    assert DictDiffer(p, q).changed() == set()


def test_change_params():
    from core import set_params, convert_xml_dict
    params = {
        'Connection': {
            '_list': {
                    3: {'In': u'adex E', 'Out': u'adex I',
                        'content': u'-1000 100. 0001'}
           }
        },
        't_end': u'1',
        'Algorithm': {
            '_list': {
                2: {'expression': u'1.',
                    'name': u'Exc Input',
                    'type': u'RateFunctor'}
            }
        }
    }
    p = convert_xml_dict(xmlpath)
    set_params(p, **params)
    assert p['Simulation']['Connections']['Connection'][3]['content'] == '-1000 100. 0001'
    assert p['Simulation']['Algorithms']['Algorithm'][2]['expression'] == '1.'
    assert p['Simulation']['SimulationRunParameter']['t_end'] == '1'


def test_no_change_params():
    from core import set_params, convert_xml_dict
    params = {
        'Connection': {
            '_list': {
                    3: {'In': u'adex E', 'Out': u'adexI',
                        'content': u'-1000 100. 0001'}
           }
       }
    }
    p = convert_xml_dict(xmlpath)
    with pytest.raises(TypeError):
        set_params(p, **params)
    params = {
        'Connection': {
            '_list': {
                    3: {'In': u'adex E', 'Out': u'adex I',
                        'content': u'nonono'}
           }
       }
    }
    p = convert_xml_dict(xmlpath)
    with pytest.raises(TypeError):
        set_params(p, **params)
    params = {
        'Connection': {
            '_list': {
                    3: {'In': u'adex E', 'Out2': u'adex I',
                        'content': u'-1000 100. 0001'}
           }
       }
    }
    p = convert_xml_dict(xmlpath)
    with pytest.raises(KeyError):
        set_params(p, **params)

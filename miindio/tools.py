import xmljson
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from pprint import pprint
import json
import os
import numpy as np
import copy
import collections
from xmldict import dict_to_xml, xml_to_dict


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


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


def read_mesh_file(filename):
    f = open(filename)
    line = f.readline()
    data = np.array([float(x) for x in line.split()[2::3]])
    if any(data < 0):
        raise ValueError('Negative density')
    return np.array(data)


def read_projection_file(filename):
    root = ET.parse(filename).getroot()
    odict = xmljson.yahoo.data(root)
    # return json.loads(json.dumps(odict))
    return root


def prettify_xml(elem):
    """Return a pretty-printed XML string for an Element, string or dict.
    """
    if isinstance(elem, (dict, collections.OrderedDict)):
        string = convert_dict_xml(elem)
    elif isinstance(elem, ET.ElementTree):
        string = ET.tostring(elem, 'utf-8')
    elif isinstance(elem, str):
        string = elem
    else:
        raise TypeError('type {} not recognized.'.format(type(elem)))
    reparsed = minidom.parseString(string)
    return reparsed.toprettyxml(indent="\t")


def dump_xml(params, fpathout):
    path, fname = os.path.split(fpathout)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(fpathout, 'w') as f:
        f.write(prettify_xml(params))


def deep_update(d, other, strict=False):
    for k, v in other.items():
        d_v = d.get(k)
        if (isinstance(v, collections.Mapping) and
            isinstance(d_v, collections.Mapping)):
            deep_update(d_v, v, strict=strict)
        else:
            if strict:
                if d_v is None:
                    raise ValueError('Unable to update "{}" '.format(k) +
                                     'according to the "strict" rule.')
                if not type(d[k]) == type(v):
                    raise TypeError("Can't set type '{}' ".format(type(d[k])) +
                                    "to new type '{}' ".format(type(v)) +
                                    "according to the 'strict' rule.")
            d[k] = copy.deepcopy(v)


def dictify(d):
    assert isinstance(d, dict)
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = {i: ii for i, ii in enumerate(v)}
        if isinstance(d[k], dict):
            d[k] = dictify(d[k])
    return d


def listify(d):
    if isdictlist(d):
        d = [d[k] for k in sorted(d.keys())]
    if isinstance(d, dict):
        for key, val in d.items():
            d[key] = listify(val)
    if isinstance(d, list):
        for i, c in enumerate(d):
            d[i] = listify(c)
    return d


def map_key(dic, key, path=None):
    path = path or []
    if isinstance(dic, collections.Mapping):
        for k, v in dic.items():
            local_path = path[:]
            local_path.append(k)
            for b in map_key(v, key, local_path):
                 yield b
    if len(path) == 0:
        pass
    elif path[-1] == key:
        yield path, dic


def pack_list_dict(parts):
    if len(parts) == 1:
        return parts[0]
    elif len(parts) > 1:
        return {parts[0]: pack_list_dict(parts[1:])}
    return parts


def set_params(params, **kwargs):
    if not kwargs:
        return
    dictify(params)
    for key, val in kwargs.items():
        mapping = list(map_key(params, key))
        if len(mapping) == 0:
            raise ValueError('Unable to map instance of "{}", '.format(key))
        if len(mapping) > 1:
            print(mapping)
            raise ValueError('Found multiple instances of "{}", '.format(key) +
                             'mapping must be unique')
        path, old_vals = mapping[0]
        if isdictlist(old_vals) and not isinstance(val, list) and not isdictlist(val):
            if all(set(val.keys()) != set(ov.keys()) for ov in old_vals):
                raise KeyError('No new keys allowed.')
            where = [DictDiffer(old_val, val).changed() == set('content')
                     for old_val in old_vals]
            if sum(where) != 1:
                raise ValueError('Unable to find matching "parameter dict".')
            old_vals[where.index(True)] = val
            val = old_vals
        if 'content' in old_vals:
            assert isinstance(old_vals, dict) and len(old_vals.keys()) == 1
            if isinstance(val, dict):
                if len(val.keys()) == 1 and 'content' in val:
                    pass
                else:
                    raise ValueError('Unable to understand value {}'.format(val))
            old_vals['content'] = val
            val = old_vals

        path.append(val)
        packed_list = pack_list_dict(path)
        deep_update(params, packed_list, strict=True)
        listify(params)


def isdictlist(val):
    if isinstance(val, dict):
        if all(isinstance(k, int) for k in val.keys()):
            idxs = sorted(val.keys())
            if all(k2 == k1 + 1 for k1, k2 in zip(idxs, idxs[1:])):
                return True
    return False



def isnumeric(val):
    assert isinstance(val, unicode)
    return remove_txt(val, ' ', '-', '.').isnumeric()


def remove_txt(txt, *args):
    for arg in args:
        txt = txt.replace(arg, '')
    return txt


def convert_xml_dict(filename):
    root = ET.parse(filename).getroot()
    odict = xml_to_dict(root)
    return odict


def convert_dict_xml(dictionary):
    elem = dict_to_xml(dictionary)
    assert len(elem) == 1
    return ET.tostring(elem[0])


def pretty_print_params(arg):
    if isinstance(arg, str):
        arg = convert_xml_dict(arg)
    else:
        assert isinstance(arg, collections.Mapping)
    pprint(json.loads(json.dumps(arg)))


def to_json(arg, fname='params.json'):
    if isinstance(arg, str):
        arg = convert_xml_dict(arg)
    else:
        assert isinstance(arg, collections.Mapping)
    with open(fname, 'w') as outfile:
        json.dump(arg, outfile,
                  sort_keys=True, indent=4)

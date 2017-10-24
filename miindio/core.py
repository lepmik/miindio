import xmljson
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from pprint import pprint
import json
import os
import glob
import ROOT
import numpy as np
import subprocess
import shutil
import copy
import pandas as pd
import collections
# From MIIND
import directories
import mesh
from ode2dsystem import Ode2DSystem


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


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
                if not d_v:
                    raise ValueError('Unable to update "{}" '.format(k) +
                                     'according to the "strict" rule.')
            d[k] = copy.deepcopy(v)


def map_key(dic, key, path=None):
    path = path or []
    if isinstance(dic, collections.Mapping):
        for k, v in dic.items():
            local_path = path[:]
            local_path.append(k)
            for b in map_key(v, key, local_path):
                 yield b
    if key in path:
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
    for key, val in kwargs.items():
        mapping = list(map_key(params, key))
        if len(mapping) == 0:
            raise ValueError('Unable to map instance of "{}", '.format(key))
        if len(mapping) > 1:
            raise ValueError('Found multiple instances of "{}", '.format(key) +
                             'mapping must be unique')
        path, old_vals = mapping[0]
        if isinstance(old_vals, list) and '_list' in val:
            assert len(val.keys()) == 1
            new_vals = val['_list']
            assert isinstance(new_vals, collections.Mapping)
            for idx in new_vals:
                # assert there are no new keys
                if set(new_vals[idx].keys()) != set(old_vals[idx].keys()):
                    raise KeyError('No new keys allowed.')
                # assert the values that are changed are numeric values (see isnumeric)
                changed_keys = [k for k, nv in new_vals[idx].items()
                                if old_vals[idx][k] != nv]
                if not  all(isnumeric(new_vals[idx][k]) for k in changed_keys):
                    bad_vals = [(k, new_vals[idx][k]) for k in changed_keys]
                    raise TypeError('Unable to change non numeric values ' +
                                    '{}'.format(bad_vals))
                old_vals[idx] = new_vals[idx]
            val = old_vals
        path.append(val)
        packed_list = pack_list_dict(path)
        deep_update(params, packed_list, strict=True)


def remove_txt(txt, *args):
    for arg in args:
        txt = txt.replace(arg, '')
    return txt


def isnumeric(val):
    assert isinstance(val, unicode)
    return remove_txt(val, ' ', '-', '.').isnumeric()


def convert_xml_dict(filename):
    root = ET.parse(filename).getroot()
    odict = xmljson.yahoo.data(root)
    return odict


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
        json.dump(prm_no_units, outfile,
                  sort_keys=True, indent=4)


def convert_dict_xml(dictionary):
    elem = xmljson.yahoo.etree(dictionary)
    assert len(elem) == 1
    return ET.tostring(elem[0])


class MiindIO:
    def __init__(self, xml_path, submit_name=None, cwd=None,
                 MIIND_BUILD_PATH=None):
        self.cwd = cwd or os.getcwd()
        self.xml_path = os.path.abspath(os.path.join(self.cwd, xml_path))
        assert os.path.exists(self.xml_path)
        if MIIND_BUILD_PATH is not None:
            MIIND_BUILD_PATH = os.path.abspath(MIIND_BUILD_PATH)
            assert os.path.exists(MIIND_BUILD_PATH)
        self.xml_location, self.xml_fname = os.path.split(self.xml_path)
        xml_base_fname, _ = os.path.splitext(self.xml_fname)
        self.submit_name = submit_name or xml_base_fname
        self.output_directory = os.path.join(self.xml_location, submit_name,
                                             xml_base_fname)
        self.miind_executable = xml_base_fname
        self.params = convert_xml_dict(self.xml_path)
        algorithm = self.params['Simulation']['Algorithms']['Algorithm']
        mesh_algo = [d for d in algorithm if d['type'] == 'MeshAlgorithm'][0]
        self.model_fname = os.path.split(mesh_algo['modelfile'])[-1]
        self.mat_fnames = mesh_algo['MatrixFile']
        self.mesh_basename = os.path.splitext(self.model_fname)[0]
        self.mesh_fname = self.mesh_basename + '.mesh'
        self.mesh_pathname = os.path.join(self.xml_location, self.mesh_fname)
        self.WITH_STATE = self.params['Simulation']['SimulationIO']['WithState'] == 'TRUE'
        self.simulation_name = self.params['Simulation']['SimulationIO']['SimulationName']
        self.root_path = os.path.join(self.output_directory,
                                      self.simulation_name + '_0.root')
        if MIIND_BUILD_PATH is None:
            print('Trying to find the MIIND build path.')
            srch = 'miind/build'
            path = [n for n in os.environ["PATH"].split(':') if srch in n]
            if len(path) > 0:
                path = path[0]
            else:
                raise IOError('Unable to find MIIND build path, looking for' +
                              '"{}" in the PATH variable.'.format(srch))
            idx = path.index(srch) + len(srch)
            path = path[:idx]
            self.MIIND_APPS = os.path.join(path, 'apps')
        else:
            self.MIIND_APPS = os.path.join(MIIND_BUILD_PATH, 'apps')
        assert os.path.exists(self.MIIND_APPS)

    def convert_root(self, save=True):
        f = ROOT.TFile(self.root_path)
        keys = [key.GetName() for key in list(f.GetListOfKeys())]
        graphs = {key: f.Get(key) for key in keys
                  if isinstance(f.Get(key), ROOT.TGraph)}
        rdata = {}
        for key, g in graphs.iteritems():
            x = g.GetX()
            y = g.GetY()
            N = g.GetN()
            x.SetSize(N)
            y.SetSize(N)
            xa = np.array(x, copy=True)
            ya = np.array(y, copy=True)
            rdata[key] = np.hstack([xa.reshape(len(xa), 1),
                                    ya.reshape(len(ya), 1)])
        data = {'x': {'_'.join(key.split('_')[:2]): list() for key in rdata.keys()},
                'y': {'_'.join(key.split('_')[:2]): list() for key in rdata.keys()}}
        keys = ['x', 'y']
        for i, xy in enumerate(keys):
            pd_idx = {'_'.join(key.split('_')[:2]): list() for key in rdata.keys()
                      if len(key.split('_')) == 3}
            for key, val in rdata.iteritems():
                sp = key.split('_')
                skey = '_'.join(key.split('_')[:2])
                if len(sp) == 3:
                    sval = float(sp[-1])
                    pd_idx[skey].append(sval)
                    data[xy][skey].append(val[:,i])
                else:
                    data[xy][skey].append(val[:,i])
            # Convert to pandas rdataFrame
            for key, val in data[xy].iteritems():
                if key in pd_idx:
                    data[xy][key] = pd.DataFrame(val, index=pd_idx[key]).sort_index()
                else:
                    data[xy][key] = pd.DataFrame(val).T
        dname = os.path.join(os.path.splitext(self.root_path)[0]+'.npz')
        if save:
            np.savez(dname, data=data)
        print 'Extracted %i graphs from root file' % len(data.keys())
        return data

    def convert_mesh(self, save=True):
        if not self.WITH_STATE:
            raise ValueError('State is not saved.')
        if not os.path.exists(self.mesh_fname):
            raise ValueError('Need ".mesh" file.')
        proj_pathname = os.path.join(
            self.xml_location, self.mesh_basename + '.projection')
        if not os.path.exists(proj_pathname):
            projection_exe = os.path.join(self.MIIND_APPS, 'Projection',
                                          'Projection')
            subprocess.call([projection_exe, self.mesh_pathname],
                            cwd=self.xml_location)
            vmin, vmax, vn, wmin, wmax, wn = input('Input "vmin, vmax, vn, wmin, wmax, wn"')
            subprocess.call([projection_exe, self.mesh_pathname, vmin, vmax, vn, wmin, wmax, wn], cwd=self.cwd)
        projection = read_projection_file(proj_pathname)
        return projection
        fnames = glob.glob(os.path.join(self.output_directory,
                                        self.model_fname + '_mesh', 'mesh*'))

        base = os.path.join(self.output_directory, self.model_fname.split('.')[0])
        m = mesh.Mesh(None)
        m.FromXML(base + '.mesh.bak')
        ode_sys = Ode2DSystem(m, [], [])

        def get_mesh_time(path):
            fname = os.path.split(path)[-1]
            return float(fname.split('_')[2])
        densities = []
        times = []
        for fname in sorted(fnames, key=get_mesh_time):
            density = read_mesh_file(fname)
            times.append(get_mesh_time(fname))
            densities.append(density)
            for i, cells in enumerate(m.cells):
                for j, cell in enumerate(cells):
                    print(cell)
                    print(ode_sys.map(i,j))
                    print(density[ode_sys.map(i,j)])
                    raise ValueError

    def generate(self, overwrite=False, **kwargs):
        set_params(self.params, **kwargs)
        shutil.copyfile(self.xml_path, self.xml_path + '.bak')
        dump_xml(self.params, self.xml_path)
        if os.path.exists(self.output_directory) and overwrite:
            shutil.rmtree(self.output_directory)
        with cd(self.cwd):
            directories.add_executable(self.submit_name, [self.xml_path], '')
        fnames = os.listdir(self.output_directory)
        if 'CMakeLists.txt' in fnames:
            subprocess.call(['cmake', '-DCMAKE_BUILD_TYPE=Release',
                             '-DCMAKE_CXX_FLAGS=-fext-numeric-literals'],
                             cwd=self.output_directory)
            subprocess.call(['make'], cwd=self.output_directory)
            subprocess.call(['cp', self.xml_path, '.'],
                            cwd=self.output_directory)

    def run(self):
        subprocess.call('./' + self.miind_executable, cwd=self.output_directory)

if __name__ == '__main__':
    io = MiindIO(xml_path='cond.xml', submit_name='cond')
    io.convert_mesh()

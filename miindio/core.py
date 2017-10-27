import os
import glob
import ROOT
import numpy as np
import subprocess
import shutil
import copy
import pandas as pd
import collections
from xmldict import dict_to_xml, xml_to_dict
from tools import *
# From MIIND
import directories
import mesh
from ode2dsystem import Ode2DSystem


class MiindIO:
    def __init__(self, xml_path, submit_name=None,
                 MIIND_BUILD_PATH=None):
        self.xml_path = os.path.abspath(xml_path)
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
        self.load_xml()
        algorithm = self.params['Simulation']['Algorithms']['Algorithm']
        mesh_algo = [d for d in algorithm if d['type'] == 'MeshAlgorithm'][0]
        self.model_fname = os.path.split(mesh_algo['modelfile'])[-1]
        self.mat_fnames = mesh_algo['MatrixFile']
        self.mesh_basename = os.path.splitext(self.model_fname)[0]
        self.mesh_fname = self.mesh_basename + '.mesh'
        self.mesh_pathname = os.path.join(self.xml_location, self.mesh_fname)
        simio = self.params['Simulation']['SimulationIO']
        self.WITH_STATE = simio['WithState']['content']
        self.simulation_name = simio['SimulationName']['content']
        self.root_path = os.path.join(self.output_directory,
                                      self.simulation_name + '_0.root')
        if MIIND_BUILD_PATH is None:
            srch = 'miind/build'
            path = [n for n in os.environ["PATH"].split(':') if srch in n]
            if len(path) > 0:
                path = path[0]
            else:
                raise IOError('Unable to find MIIND build path, looking for' +
                              '"{}" in the PATH variable.'.format(srch))
            idx = path.index(srch) + len(srch)
            path = path[:idx]
            print('Found the MIIND build path at {}.'.format(path))
            self.MIIND_APPS = os.path.join(path, 'apps')
        else:
            self.MIIND_APPS = os.path.join(MIIND_BUILD_PATH, 'apps')
        assert os.path.exists(self.MIIND_APPS)

    def convert_root(self, verbose=False):
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
        if verbose:
            print 'Extracted %i graphs from root file' % len(data.keys())
        return data

    @property
    def data(self):
        if not hasattr(self, '_data'):
            self._data = self.convert_root()
            xmlpath = os.path.join(self.output_directory, self.xml_fname)
            self._data['params'] = convert_xml_dict(xmlpath)
            if dict_changed(self._data['params'], self.params):
                print('WARNING: Data parameters and self parameters are ' +
                          'not equal. Use "data["params"]"')
        return self._data

    def set_params(self, **kwargs):
        set_params(self.params, **kwargs)

    def save_data(self):
        dname = os.path.join(os.path.splitext(self.root_path)[0]+'.npz')
        np.savez(dname, data=data)

    def dump_xml(self):
        dump_xml(self.params, self.xml_path)

    def load_xml(self):
        self.params = convert_xml_dict(self.xml_path)

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
            subprocess.call([projection_exe, self.mesh_pathname, vmin, vmax, vn, wmin, wmax, wn], cwd=self.xml_location)
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
        self.load_xml()
        shutil.copyfile(self.xml_path, self.xml_path + '.bak')
        self.set_params(**kwargs)
        self.dump_xml()
        if os.path.exists(self.output_directory) and overwrite:
            shutil.rmtree(self.output_directory)
        with cd(self.xml_location):
            directories.add_executable(self.submit_name, [self.xml_path], '')
        fnames = os.listdir(self.output_directory)
        if 'CMakeLists.txt' in fnames:
            subprocess.call(['cmake', '-DCMAKE_BUILD_TYPE=Release',
                             '-DCMAKE_CXX_FLAGS=-fext-numeric-literals'],
                             cwd=self.output_directory)
            subprocess.call(['make'], cwd=self.output_directory)
            shutil.move(self.xml_path,
                        os.path.join(self.output_directory, self.xml_fname))
            shutil.move(self.xml_path + '.bak', self.xml_path)

    def run(self):
        subprocess.call('./' + self.miind_executable, cwd=self.output_directory)

if __name__ == '__main__':
    io = MiindIO(xml_path='cond.xml', submit_name='cond')
    io.convert_mesh()

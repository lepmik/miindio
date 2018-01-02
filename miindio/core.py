import os
import os.path as op
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
import hashlib
from density import Density, Marginal
# From MIIND
import directories


class MiindIO:
    def __init__(self, xml_path, submit_name=None,
                 MIIND_BUILD_PATH=None, **kwargs):
        xml_path = op.abspath(xml_path)
        # load current
        self.params = convert_xml_dict(xml_path)
        # set new params
        if kwargs:
            self.set_params(**kwargs)
            # generate sha based on parameters
            self.xml_path = xml_path.replace('.xml', self.sha + '.xml')
            # dump to unique self.xml_path so miind will use new params
            self.dump_xml()
        else:
            self.xml_path = xml_path
        assert op.exists(self.xml_path)
        if MIIND_BUILD_PATH is not None:
            MIIND_BUILD_PATH = op.abspath(MIIND_BUILD_PATH)
            assert op.exists(MIIND_BUILD_PATH)
        self.xml_location, self.xml_fname = op.split(self.xml_path)
        xml_base_fname, _ = op.splitext(self.xml_fname)
        self.submit_name = submit_name or xml_base_fname
        self.output_directory = op.join(
            self.xml_location, self.submit_name, xml_base_fname)
        self.miind_executable = xml_base_fname
        simpar = self.params['Simulation']
        self.modelfiles = [m.get('modelfile')
                           for m in simpar['Algorithms']['Algorithm']
                           if m.get('modelfile') is not None]
        simio = simpar['SimulationIO']
        self.WITH_STATE = simio['WithState']['content']
        self.simulation_name = simio['SimulationName']['content']
        self.root_path = op.join(self.output_directory,
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
            self.MIIND_APPS = op.join(path, 'apps')
        else:
            self.MIIND_APPS = op.join(MIIND_BUILD_PATH, 'apps')
        assert op.exists(self.MIIND_APPS)
        if self.WITH_STATE:
            modnames = [split_fname(mn, '.model')[0] for mn in self.modelfiles]
            self.density = {mn: Density(self, mn) for mn in modnames}
            self.marginal = {mn: Marginal(self, mn) for mn in modnames}

    @property
    def sha(self):
        assert hasattr(self, 'params')
        par_str = json.dumps(self.params)
        sha = hashlib.sha1(par_str).hexdigest()
        return sha

    @property
    def rates(self):
        if hasattr(self, '_rates'):
            return self._rates
        fnameout = op.join(self.output_directory,
                                 self.simulation_name + '_rates.npz')
        if op.exists(fnameout):
            return np.load(fnameout)['data'][()]
        f = ROOT.TFile(self.root_path)
        keys = [key.GetName() for key in list(f.GetListOfKeys())]
        graphs = {key: f.Get(key) for key in keys
                  if isinstance(f.Get(key), ROOT.TGraph)}
        rdata = {}
        for key, g in graphs.iteritems():
            x, y, N = g.GetX(), g.GetY(), g.GetN()
            x.SetSize(N)
            y.SetSize(N)
            xa = np.array(x, copy=True)
            ya = np.array(y, copy=True)
            rdata[key] = np.hstack([xa.reshape(len(xa), 1),
                                    ya.reshape(len(ya), 1)])
        _rates = {'x': {'_'.join(key.split('_')[:2]): list() for key in rdata.keys()},
                'y': {'_'.join(key.split('_')[:2]): list() for key in rdata.keys()}}
        keys = ['x', 'y']
        for i, xy in enumerate(keys):
            for key, val in rdata.iteritems():
                sp = key.split('_')
                skey = '_'.join(key.split('_')[:2])
                if len(sp) == 3:
                    continue
                else:
                    _rates[xy][skey].append(val[:, i])
            # Convert to np array
            for key, val in _rates[xy].iteritems():
                _rates[xy][key] = np.array(val).flatten()[2::2] # TODO HACK TODO why every other here??? bug in miind??
        print 'Extracted %i graphs from root file' % len(_rates['x'].keys())
        self._rates = _rates
        np.savez(fnameout, data=_rates)
        return _rates

    @property
    def run_exists(self):
        '''
        checks if this particular
        '''
        xmlpath = op.join(self.output_directory, self.xml_fname)
        modelfiles = [op.join(self.output_directory, m)
                      for m in self.modelfiles]
        if not op.exists(xmlpath):
            return False
        old_params = convert_xml_dict(xmlpath)
        if not op.exists(self.root_path):
            return False
        if self.WITH_STATE:
            for p in modelfiles:
                if not op.exists(p + '_mesh'):
                    return False
                if len(os.listdir(p + '_mesh')) == 0:
                    return False
        return dict_changed(old_params, self.params) == set()

    def set_params(self, **kwargs):
        if not hasattr(self, 'params'):
            self.load_xml()
        set_params(self.params, **kwargs)

    def dump_xml(self):
        dump_xml(self.params, self.xml_path)

    def load_xml(self):
        self.params = convert_xml_dict(self.xml_path)

    def generate(self, **kwargs):
        if op.exists(self.output_directory):
            shutil.rmtree(self.output_directory)
        with cd(self.xml_location):
            directories.add_executable(self.submit_name, [self.xml_path], '')
        fnames = os.listdir(self.output_directory)
        if 'CMakeLists.txt' in fnames:
            subprocess.call(['cmake', '-DCMAKE_BUILD_TYPE=Release',
                             '-DCMAKE_CXX_FLAGS=-fext-numeric-literals'],
                             cwd=self.output_directory)
            subprocess.call(['make'], cwd=self.output_directory)
            shutil.copyfile(self.xml_path, op.join(self.output_directory,
                                                        self.xml_fname))

    def run(self):
        subprocess.call('./' + self.miind_executable, cwd=self.output_directory)

if __name__ == '__main__':
    io = MiindIO(xml_path='cond.xml', submit_name='cond')
    io.get_marginals()

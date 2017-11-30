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
import hashlib
# From MIIND
import directories
import mesh
from ode2dsystem import Ode2DSystem


class MiindIO:
    def __init__(self, xml_path, submit_name=None,
                 MIIND_BUILD_PATH=None, **kwargs):
        xml_path = os.path.abspath(xml_path)
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
        assert os.path.exists(self.xml_path)
        if MIIND_BUILD_PATH is not None:
            MIIND_BUILD_PATH = os.path.abspath(MIIND_BUILD_PATH)
            assert os.path.exists(MIIND_BUILD_PATH)
        self.xml_location, self.xml_fname = os.path.split(self.xml_path)
        xml_base_fname, _ = os.path.splitext(self.xml_fname)
        self.submit_name = submit_name or xml_base_fname
        self.output_directory = os.path.join(
            self.xml_location, self.submit_name, xml_base_fname)
        self.miind_executable = xml_base_fname
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

    @property
    def sha(self):
        assert hasattr(self, 'params')
        par_str = json.dumps(self.params)
        sha = hashlib.sha1(par_str).hexdigest()
        return sha

    def get_rates(self):
        fnameout = os.path.join(self.output_directory,
                                 self.simulation_name + '_rates.npz')
        if hasattr(self, '_rates'):
            return self._rates
        if os.path.exists(fnameout):
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
        xmlpath = os.path.join(self.output_directory, self.xml_fname)
        if not os.path.exists(xmlpath):
            return False
        old_params = convert_xml_dict(xmlpath)
        if not os.path.exists(self.root_path):
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

    def get_marginal_density(self, basename, vn=100, wn=100, timestep=None,
                             time=None, force=False):
        if not self.WITH_STATE:
            raise ValueError('State is not saved.')
        fnameout = os.path.join(self.output_directory,
                                basename + '_marginal_density.npz')
        if not force:
            if hasattr(self, '_marginal_density_' + basename):
                return getattr(self, '_marginal_density_' + basename)
            if os.path.exists(fnameout):
                return np.load(fnameout)['data'][()]
        modelname = basename + '.model'
        modelpath = os.path.join(self.xml_location, modelname)
        assert os.path.exists(modelpath)
        meshpath = extract_mesh(modelpath)
        proj = self.read_projection(basename, vn, wn)
        fnames = glob.glob(os.path.join(self.output_directory,
                                        modelname + '_mesh', 'mesh*'))
        if len(fnames) == 0:
            raise ValueError('No density output found for {}'.format(basename))
        m = mesh.Mesh(None)
        m.FromXML(meshpath)
        ode_sys = Ode2DSystem(m, [], [])

        def get_density_time(path):
            fname = os.path.split(path)[-1]
            return float(fname.split('_')[2])

        def get_scaling(proj):
            scale = []
            a = 0
            for marg in proj.split(';'):
                if len(marg) == 0:
                    continue
                jj, dd = marg.split(',')
                a += float(dd)
                scale.append((int(jj), float(dd)))
            assert a - 1 < 1e-7, a
            return scale

        fnames = sorted(fnames, key=get_density_time)
        times = [get_density_time(f) for f in fnames]
        if timestep is not None:
            assert time is None
            fnames = fnames[::timestep]
            times = times[::timestep]
        if time is not None:
            assert timestep is None
            if time == 'end':
                time = times[-1]
            elif not isinstance(time, float):
                raise TypeError('"time" must be a float or the string "end"')
            fnames = [fnames[times.index(time)]]
            times = [time]
        vs = np.zeros((len(fnames), proj['N_V']))
        ws = np.zeros((len(fnames), proj['N_W']))

        print('Computing marginals.')
        for ii, fname in enumerate(fnames):
            try:
                density = read_density(fname)
            except ValueError:
                print('Unable to read ' + fname)
                continue
            for idx, (i, j) in enumerate(proj['coords']):
                cell_dens = density[ode_sys.map(i, j)]
                if cell_dens < 1e-15:
                    continue
                vbins = proj['vbins'][idx]
                wbins = proj['wbins'][idx]
                for jj, dd in get_scaling(vbins):
                    vs[ii, jj] += cell_dens * dd
                for jj, dd in get_scaling(wbins):
                    ws[ii, jj] += cell_dens * dd
        dv = abs(proj['V_max'] - proj['V_min']) / float(proj['N_V'])
        bins_v = np.linspace(proj['V_min'], proj['V_max'], proj['N_V'])
        dw = abs(proj['W_max'] - proj['W_min']) / float(proj['N_W'])
        bins_w = np.linspace(proj['W_min'], proj['W_max'], proj['N_W'])
        # Normalize to density
        for db, n_ in zip([dv, dw], [vs, ws]):
            for i in range(len(fnames)):
                n_[i, :] = n_[i, :] / db / n_[i, :].sum()
        data = {'v': vs, 'w': ws, 'bins_v': bins_v,
                'bins_w': bins_w, 'times': times}
        setattr(self, '_marginal_density_' + basename, data)
        np.savez(fnameout, data=data)
        return data

    def make_projection_file(self, basename, vn, wn):
        projection_exe = os.path.join(self.MIIND_APPS, 'Projection',
                                      'Projection')
        out = subprocess.check_output(
            [projection_exe, basename + '.mesh.bak'],
             cwd=self.xml_location)
        vmax, wmax = np.ceil(np.array(out.split('\n')[3].split(' ')[2:],
                                      dtype=float)).astype(int)
        vmin, wmin = np.floor(np.array(out.split('\n')[4].split(' ')[2:],
                                      dtype=float)).astype(int)
        cmd = [projection_exe, basename + '.mesh.bak', vmin, vmax,
               vn, wmin, wmax, wn]
        subprocess.call([str(c) for c in cmd], cwd=self.xml_location)

    def read_projection(self, basename, vn, wn):
        proj_pathname = os.path.join(
            self.xml_location, basename + '.projection')
        if not os.path.exists(proj_pathname):
            print('No projection file found, generating...')
            self.make_projection_file(basename, vn, wn)
        proj = xml_to_dict(ET.parse(proj_pathname).getroot(),
                           text_content=None)
        if (proj['Projection']['W_limit']['N_W'] != wn or
                proj['Projection']['V_limit']['N_V'] != vn):
            print('New N in bins, generating projection file...')
            self.make_projection_file(basename, vn, wn)
            proj = xml_to_dict(ET.parse(proj_pathname).getroot(),
                               text_content=None)
        # TODO reading below not necessary, when marc makes a proper xml
        coords = []
        vbins, wbins = [], []
        with open(proj_pathname, 'r') as f:
            read = False
            for l in f:
                l = l.strip()
                if l == '</W_limit>':
                    read = True
                    continue
                if read:
                    if l == '</Projection>':
                        continue
                    s1 = l.split(',')
                    s2 = s1[1].split(';')
                    coords.append((int(s1[0]), int(s2[0])))
                    vbins.append(remove_txt(l.split('vbins')[1], '<', '>', '/'))
                    wbins.append(remove_txt(l.split('wbins')[1], '<', '>', '/'))
        assert proj['Projection']['vbins'] == vbins
        assert proj['Projection']['wbins'] == wbins
        assert len(set(coords)) == len(coords)

        return {
            'coords': coords,
            'vbins': vbins,
            'wbins': wbins,
            'V_min': proj['Projection']['V_limit']['V_min'],
            'V_max': proj['Projection']['V_limit']['V_max'],
            'N_V': proj['Projection']['V_limit']['N_V'],
            'W_min': proj['Projection']['W_limit']['W_min'],
            'W_max': proj['Projection']['W_limit']['W_max'],
            'N_W': proj['Projection']['W_limit']['N_W'],
        }

    def plot_marginal_density(self, basename, **args):
        import matplotlib.pyplot as plt
        data = self.get_marginal_density(basename, **args)
        path = os.path.join(self.output_directory,
                            basename + '_marginal_density')
        if not os.path.exists(path):
            os.mkdir(path)
        for ii in range(len(data['times'])):
            fig, axs = plt.subplots(1, 2)
            params = {
                'ax': axs,
                'dens': [data['v'], data['w']],
                'bins': [data['bins_v'], data['bins_w']]
            }
            params = [{k: v[i] for k, v in params.items()}
                      for i in range(len(params['ax']))]
            plt.suptitle('time = {}'.format(data['times'][ii]))
            for p in params:
                p['ax'].plot(p['bins'], p['dens'][ii, :])
                fig.savefig(os.path.join(path,
                            'marginal_density_{}.png'.format(ii)))
                plt.close(fig)

    def plot_density(self, basename, colorbar=[1e-6,1.,100]):
        import visualize
        path = os.path.join(self.output_directory,
                            basename + '_density')
        if not os.path.exists(path):
            os.mkdir(path)
        modelname = basename + '.model'
        fnames = glob.glob(os.path.join(self.output_directory,
                                        modelname + '_mesh', 'mesh*'))

        def get_density_time(path):
            fname = os.path.split(path)[-1]
            return float(fname.split('_')[2])

        fnames = sorted(fnames, key=get_density_time)
        m = visualize.ModelVisualizer(modelname)
        for i, fpath in enumerate(fnames):
            time = get_density_time(fpath)
            fname = os.path.split(fpath)[-1]
            m.showfile(fpath,
                       pdfname=os.path.join(path, '%i_'%i + fname),
                       runningtext='t = %f'%time,
                       colorlegend=colorbar)

    def generate(self, **kwargs):
        if os.path.exists(self.output_directory):
            shutil.rmtree(self.output_directory)
        with cd(self.xml_location):
            directories.add_executable(self.submit_name, [self.xml_path], '')
        fnames = os.listdir(self.output_directory)
        if 'CMakeLists.txt' in fnames:
            subprocess.call(['cmake', '-DCMAKE_BUILD_TYPE=Release',
                             '-DCMAKE_CXX_FLAGS=-fext-numeric-literals'],
                             cwd=self.output_directory)
            subprocess.call(['make'], cwd=self.output_directory)
            shutil.copyfile(self.xml_path, os.path.join(self.output_directory,
                                                        self.xml_fname))

    def run(self):
        subprocess.call('./' + self.miind_executable, cwd=self.output_directory)

if __name__ == '__main__':
    io = MiindIO(xml_path='cond.xml', submit_name='cond')
    io.get_marginals()

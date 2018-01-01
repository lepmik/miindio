import os.path as op
from tools import *
import mesh as meshmod
import glob
import subprocess
import time
import matplotlib.pyplot as plt
from collections import OrderedDict as odict
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from matplotlib.collections import PatchCollection


def replace(value, string, *args):
    for a in args:
        value = value.replace(a, string)
    return value


def find_density_fnames(modelfname, directory):
    fnames = glob.glob(op.join(directory, modelfname + '_mesh', 'mesh*'))
    if len(fnames) == 0:
        raise ValueError('No density output found for {}'.format(modelfname))

    fnames = sorted(fnames, key=get_density_time)
    times = [get_density_time(f) for f in fnames]
    return fnames, times


def read_density(filename):
    f = open(filename, 'r')
    line = f.readline().split()
    data = [float(x) for x in line[2::3]]
    coords = [(int(i), int(j)) for i, j in zip(line[::3], line[1::3])]
    return data, coords


def get_density_time(path):
    fname = op.split(path)[-1]
    return float(fname.split('_')[2])


def calc_mass(mesh, density, coords):
    masses = [mesh.cells[i][j].area * dens
              for (i, j), dens in zip(coords, density)]
    return masses


class Marginal:
    def __init__(self, io, modelname, vn=100, wn=100):
        self.io = io
        self.modelname, self.modelfname = split_fname(modelname, '.model')
        self.path = op.join(self.io.output_directory,
                            self.modelname + '_marginal_density')
        self.data_path = op.join(self.io.output_directory, 'marginal_density.npz')
        if not op.exists(self.path):
            os.mkdir(self.path)
        self.fnames, self.times = find_density_fnames(
            self.modelfname, self.io.output_directory)
        self.projfname = self.modelfname.replace('.model', '.projection')
        self.vn, self.wn = vn, wn

    def __getitem__(self, name):
        return self.density[name]

    @property
    def mesh(self):
        if not hasattr(self, '_mesh'):
            mesh = meshmod.Mesh(None)
            mesh.FromXML(self.modelfname)
            self._mesh = mesh
        return self._mesh

    @property
    def density(self):
        self.read_projection()
        if op.exists(self.data_path) and not self.new_projection:
            load_data = np.load(self.data_path)['data'][()]
            if self.modelname in load_data:
                print('Loading marginal density of {}'.format(self.modelfname) +
                      ' from file.')
                return load_data[self.modelname]
        print('Calculating marginal density of {}'.format(self.modelfname) +
              ', this can take a while.')
        tm = time.time()
        v = np.zeros((len(fnames), self.projection['N_V']))
        w = np.zeros((len(fnames), self.projection['N_W']))
        masses, coords_ = [], None
        for ii, fname in enumerate(self.fnames):
            density, coords = read_density(fname)
            if coords_ is None:
                coords_ = coords
            else:
                assert coords == coords_
            masses.append(calc_mass(self.mesh, density, coords))
        masses = np.vstack(masses)
        assert masses.shape[0] == len(self.fnames)
        v, w, bins_v, bins_w = self.calc_marginal_density(
            v, w, masses, coords, self.projection)
        data = {'v': v, 'w': w, 'bins_v': bins_v,
                'bins_w': bins_w, 'times': times}
        print('Calculating marginal density of {}'.format(self.modelfname) +
              ', took {} s.'.format(time.time() - tm))
        if op.exists(self.data_path):
            other = np.load(self.data_path)['data'][()]
            save_data = other.update({self.modelname: data})
        else:
            save_data = {self.modelname: data}
        np.savez(self.data_path, data=save_data)
        return data

    def calc_marginal_density(self, v, w, masses, coords, proj):

        def scale(var, proj, mass):
            bins = [marg.split(',') for marg in proj.split(';')
                    if len(marg) > 0]
            for jj, dd in bins:
                var[:, int(jj)] += mass * float(dd)
            return var
        for trans in proj['transitions']:
            i, j = [int(a) for a in trans['coordinates'].split(',')]
            cell_mass = masses[:, coords.index((i, j))]
            if np.all(cell_mass < 1e-15):
                continue
            v = scale(v, trans['vbins'], cell_mass)
            w = scale(w, trans['wbins'], cell_mass)
        dv = abs(proj['V_max'] - proj['V_min']) / float(proj['N_V'])
        dw = abs(proj['W_max'] - proj['W_min']) / float(proj['N_W'])
        for idx in range(v.shape[0]):
            v[idx] = v[idx] / dv / v[idx].sum()
            w[idx] = w[idx] / dw / w[idx].sum()
        bins_v = np.linspace(proj['V_min'], proj['V_max'], proj['N_V'])
        bins_w = np.linspace(proj['W_min'], proj['W_max'], proj['N_W'])
        return v, w, bins_v, bins_w

    def make_projection_file(self):
        projection_exe = op.join(self.io.MIIND_APPS, 'Projection', 'Projection')
        out = subprocess.check_output(
          [projection_exe, self.modelfname], cwd=self.io.xml_location)
        vmax, wmax = np.ceil(np.array(out.split('\n')[3].split(' ')[2:],
                                    dtype=float)).astype(int)
        vmin, wmin = np.floor(np.array(out.split('\n')[4].split(' ')[2:],
                                    dtype=float)).astype(int)
        cmd = [projection_exe, self.modelfname, vmin, vmax,
               self.vn, wmin, wmax, self.wn]
        subprocess.call([str(c) for c in cmd], cwd=self.io.xml_location)

    def read_projection(self):
        proj_pathname = op.join(self.io.xml_location, self.projfname)
        self.new_projection = False
        if not op.exists(proj_pathname):
            print('No projection file found, generating...')
            self.make_projection_file(self.modelfname, self.vn, self.wn)
            self.new_projection = True
        if not hasattr(self, 'projection'):
            self._proj = xml_to_dict(ET.parse(proj_pathname).getroot(),
                                      text_content=None)['Projection']
        if (self._proj['W_limit']['N_W'] != self.wn or
            self._proj['V_limit']['N_V'] != self.vn):
            print('New N in bins, generating projection file...')
            self.make_projection_file(self.modelfname, self.vn, self.wn)
            self._proj = xml_to_dict(ET.parse(proj_pathname).getroot(),
                                      text_content=None)['Projection']
            self.new_projection = True
        self.projection =  {
            'transitions': self._proj['transitions']['cell'],
            'V_min': self._proj['V_limit']['V_min'],
            'V_max': self._proj['V_limit']['V_max'],
            'N_V': self._proj['V_limit']['N_V'],
            'W_min': self._proj['W_limit']['W_min'],
            'W_max': self._proj['W_limit']['W_max'],
            'N_W': self._proj['W_limit']['N_W'],
        }

    def plot_marginal_density(self):

        for ii in range(len(self['times'])):
            fig, axs = plt.subplots(1, 2)
            params = {
                'ax': axs,
                'dens': [self['v'], self['w']],
                'bins': [self['bins_v'], self['bins_w']]
            }
            params = [{k: v[i] for k, v in params.items()}
                      for i in range(len(params['ax']))]
            plt.suptitle('time = {}'.format(self['times'][ii]))
            for p in params:
                p['ax'].plot(p['bins'], p['dens'][ii, :])
                figname = op.join(path,
                                  '{}_'.format(ii) +
                                  '{}.png'.format(self['times'][ii]))
                fig.savefig(figname, res=300, bbox_inches='tight')
                plt.close(fig)


class Density:
    def __init__(self, io, modelname):
        self.io = io
        self.modelname, self.modelfname = split_fname(modelname, '.model')
        self.path = op.join(self.io.output_directory,
                            self.modelname + '_density')
        if not op.exists(self.path):
            os.mkdir(self.path)
        self.fnames, self.times = find_density_fnames(
            self.modelfname, self.io.output_directory)

    @property
    def mesh(self):
        if not hasattr(self, '_mesh'):
            mesh = meshmod.Mesh(None)
            mesh.FromXML(self.modelfname)
            self._mesh = mesh
        return self._mesh

    @property
    def polygons(self):
        if not hasattr(self, '_polygons'):
            self._polygons = odict(
                ((i, j),
                Polygon([(float(x), float(y))
                         for x, y in zip(cell.vs, cell.ws)]))
                for i, cells in enumerate(self.mesh.cells)
                for j, cell in enumerate(cells)
            )
        return self._polygons

    @property
    def patches(self):
        if not hasattr(self, '_patches'):
            self._patches = [
                PolygonPatch(polygon)
                for polygon in self.polygons.values()
            ]
        return self._patches

    def plot_mesh(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        md = self.mesh.dimensions()
        p = PatchCollection(self.patches, alpha=1, edgecolors='k',
                            facecolors='w')
        ax.add_collection(p)
        ax.set_xlim(md[0])
        ax.set_ylim(md[1])
        aspect = (md[0][1] - md[0][0]) / (md[1][1] - md[1][0])
        ax.set_aspect(aspect)
        return ax

    def plot_density(self, fname, colorbar=None, cmap='inferno', ax=None,
                     save=True, ext='.png'):
        if not ext.startswith('.'):
            ext = '.' + ext
        if ax is None:
            fig, ax = plt.subplots()
        time = get_density_time(fname)
        idx = self.times.index(time)
        md = self.mesh.dimensions()
        poly_coords = list(self.polygons.keys())
        ax.set_xlim(md[0])
        ax.set_ylim(md[1])
        aspect = (md[0][1] - md[0][0]) / (md[1][1] - md[1][0])
        ax.set_aspect(aspect)
        p = PatchCollection(self.patches, cmap=cmap)
        density, coords = read_density(fname)
        sort_idx = sorted(range(len(coords)), key=coords.__getitem__)
        coords = [coords[i] for i in sort_idx]
        density = [density[i] for i in sort_idx]
        assert coords == poly_coords
        p.set_array(np.array(density))
        ax.add_collection(p)
        if colorbar is not None:
            plt.colorbar(p)
        if save:
            figname = op.join(
                self.path, '{}_'.format(idx) +
                '{}'.format(time)).replace('.', '-')
            plt.gcf().savefig(figname + ext, res=300, bbox_inches='tight')

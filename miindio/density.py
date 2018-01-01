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


def split_fname(fname, ext):
    if not ext.startswith('.'):
        ext = '.' + ext
    if fname.endswith(ext):
        modelname = op.splitext(fname)[0]
        modelfname = fname
    else:
        modelname = fname
        modelfname = fname + ext
    return modelname, modelfname


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
    def __init__(self, io):
        self.io = io

    def __getitem__(self, name=None):
        return self.get_marginal_densities()[name]

    def __contains__(self, name):
        return name in self.get_marginal_densities()

    def keys(self):
        return self.get_marginal_densities().keys()

    def values(self):
        return self.get_marginal_densities().values()

    def items(self):
        return self.get_marginal_densities().items()

    def get_marginal_densities(self, modelpath=None, vn=100, wn=100,
                               force=False):
        data = {}
        save = []
        fpathout = op.join(self.io.output_directory, 'marginal_density.npz')
        modelfiles = [modelpath] if modelpath is not None else self.modelfiles
        for modelfname in modelfiles:
            key = op.splitext(modelfname)[0]
            if op.exists(fpathout):
                data_ = np.load(fpathout)['data'][()]
                if key in data_:
                    print('Loading marginal density of {}'.format(modelfname) +
                          ' from file.')
                    data.update(data_)
                    save.append(False)
                    continue
            print('Calculating marginal density of {}'.format(modelfname) +
                  ', this can take a while.')
            tm = time.time()
            projfname = modelfname.replace('.model', '.projection')
            proj, mesh = self.read_projection(projfname, vn, wn)

            fnames, times = find_density_fnames(modelfname,
                                                self.io.output_directory)
            v = np.zeros((len(fnames), proj['N_V']))
            w = np.zeros((len(fnames), proj['N_W']))
            times_ = [get_density_time(f) for f in fnames]
            masses, coords_ = [], None
            for ii, fname in enumerate(fnames):
                density, coords = read_density(fname)
                if coords_ is None:
                    coords_ = coords
                else:
                    assert coords == coords_
                masses.append(calc_mass(mesh, density, coords))
            masses = np.vstack(masses)
            assert masses.shape[0] == len(fnames)
            v, w, bins_v, bins_w = self.calc_marginal_density(
                v, w, masses, coords, proj, mesh)
            data[key] = {
                'v': v, 'w': w, 'bins_v': bins_v,
                'bins_w': bins_w, 'times': times}
            print('Calculating marginal density of {}'.format(modelfname) +
                  ', took {} s.'.format(time.time() - tm))
            save.append(True)
        if any(save):
            if op.exists(fpathout):
                other = np.load(fpathout)['data'][()]
                data = other.update(data)
            np.savez(fpathout, data=data)
        return data

    def calc_marginal_density(self, v, w, masses, coords, proj, mesh):

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

    def make_projection_file(self, modelfname, vn, wn):
        projection_exe = op.join(self.io.MIIND_APPS, 'Projection', 'Projection')
        out = subprocess.check_output(
          [projection_exe, modelfname], cwd=self.io.xml_location)
        vmax, wmax = np.ceil(np.array(out.split('\n')[3].split(' ')[2:],
                                    dtype=float)).astype(int)
        vmin, wmin = np.floor(np.array(out.split('\n')[4].split(' ')[2:],
                                    dtype=float)).astype(int)
        cmd = [projection_exe, modelfname, vmin, vmax,
             vn, wmin, wmax, wn]
        subprocess.call([str(c) for c in cmd], cwd=self.io.xml_location)

    def read_projection(self, projfname, vn, wn):
        proj_pathname = op.join(self.io.xml_location, projfname)
        modelfname = projfname.replace('.projection', '.model')
        if not op.exists(proj_pathname):
            print('No projection file found, generating...')
            self.make_projection_file(modelfname, vn, wn)
        proj = xml_to_dict(ET.parse(proj_pathname).getroot(),
                         text_content=None)
        if (proj['Projection']['W_limit']['N_W'] != wn or
        proj['Projection']['V_limit']['N_V'] != vn):
            print('New N in bins, generating projection file...')
            self.make_projection_file(modelfname, vn, wn)
            proj = xml_to_dict(ET.parse(proj_pathname).getroot(),
                               text_content=None)
        mesh = meshmod.Mesh(None)
        mesh.FromXML(proj_pathname)
        result =  {
            'transitions': proj['Projection']['transitions']['cell'],
            'V_min': proj['Projection']['V_limit']['V_min'],
            'V_max': proj['Projection']['V_limit']['V_max'],
            'N_V': proj['Projection']['V_limit']['N_V'],
            'W_min': proj['Projection']['W_limit']['W_min'],
            'W_max': proj['Projection']['W_limit']['W_max'],
            'N_W': proj['Projection']['W_limit']['N_W'],
        }
        return result, mesh

    def plot_marginal_density(self, **args):
        data_ = self.get_marginal_densities(**args)
        for modelfname, data in data_.items():
            path = op.join(self.io.output_directory,
                          op.splitext(modelfname)[0] +
                          '_marginal_density')
            if not op.exists(path):
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
                    figname = op.join(path,
                                      '{}_'.format(ii) +
                                      '{}.png'.format(data['times'][ii]))
                    fig.savefig(figname, res=300, bbox_inches='tight')
                    plt.close(fig)


class Density:
    def __init__(self, io):
        self.io = io

    def mesh(self, modelname):
        modelname, modelfname = split_fname(modelname, '.model')
        if not hasattr(self, '_mesh'):
            self._mesh = {}
        if modelname not in self._mesh:
            mesh = meshmod.Mesh(None)
            mesh.FromXML(modelfname)
            self._mesh[modelname] = mesh
        return self._mesh[modelname]

    def polygons(self, modelname):
        modelname, modelfname = split_fname(modelname, '.model')
        if not hasattr(self, '_polygons'):
            self._polygons = {}
        if modelname not in self._polygons:
            self._polygons[modelname] = odict(
                ((i, j),
                Polygon([(float(x), float(y))
                         for x, y in zip(cell.vs, cell.ws)]))
                for i, cells in enumerate(self.mesh(modelname).cells)
                for j, cell in enumerate(cells)
            )
        return self._polygons[modelname]

    def patches(self, modelname):
        modelname, modelfname = split_fname(modelname, '.model')
        if not hasattr(self, '_patches'):
            self._patches = {}
        if modelname not in self._patches:
            self._patches[modelname] = [
                PolygonPatch(polygon)
                for polygon in self.polygons(modelname).values()
            ]
        return self._patches[modelname]

    def plot_mesh(self, modelname, ax=None):
        modelname, modelfname = split_fname(modelname, '.model')
        if ax is None:
            fig, ax = plt.subplots()
        mesh = self.mesh(modelname)
        md = mesh.dimensions()
        p = PatchCollection(self.patches(modelname), alpha=1, edgecolors='k',
                            facecolors='w')
        # p.set_array(np.array(colors))
        ax.add_collection(p)
        ax.set_xlim(md[0])
        ax.set_ylim(md[1])
        aspect = (md[0][1] - md[0][0]) / (md[1][1] - md[1][0])
        ax.set_aspect(aspect)
        return ax

    def plot_density(self, modelname, time=None, timestep=None, colorbar=None,
                     cmap='inferno'):
        modelname, modelfname = split_fname(modelname, '.model')
        path = op.join(self.io.output_directory,
                       op.splitext(modelfname)[0] + '_density')
        if not op.exists(path):
            os.mkdir(path)
        fnames, times = find_density_fnames(modelfname,
                                            self.io.output_directory)
        idxs = range(len(times))
        if time is not None:
            assert timestep is None
            idx = times.index(time)
            idxs, fnames, times = [idx], [fnames[idx]], [times[idx]]
        if timestep is not None:
            assert time is None
            fnames = fnames[::timestep]
            times_ = times[::timestep]
            idxs = [times.index(time) for time in times_]
            times = times_

        mesh = self.mesh(modelname)
        md = mesh.dimensions()

        poly_coords = list(self.polygons(modelname).keys())
        for fname, time, ii in zip(fnames, times, idxs):
            fig, ax = plt.subplots()
            ax.set_xlim(md[0])
            ax.set_ylim(md[1])
            aspect = (md[0][1] - md[0][0]) / (md[1][1] - md[1][0])
            ax.set_aspect(aspect)
            p = PatchCollection(self.patches(modelname), cmap=cmap)
            density, coords = read_density(fname)
            sort_idx = sorted(range(len(coords)), key=coords.__getitem__)
            coords = [coords[i] for i in sort_idx]
            density = [density[i] for i in sort_idx]
            assert coords == poly_coords
            p.set_array(np.array(density))
            ax.add_collection(p)
            figname = op.join(path, '{}_'.format(ii) + '{}.png'.format(time))
            fig.savefig(figname, res=300, bbox_inches='tight')

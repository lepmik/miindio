import os.path as op
from tools import *
import mesh as meshmod
import glob
import subprocess
import time


def replace(value, string, *args):
    for a in args:
        value = value.replace(a, string)
    return value


class Density:
    def __init__(self, io):
        self.__dict__.update(io.__dict__)

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
        fpathout = op.join(self.output_directory, 'marginal_density.npz')
        modelfiles = [modelpath] if modelpath is not None else self.modelfiles
        for modelfname in modelfiles:
            key = modelfname.replace('.model', '')
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

            fnames, times = self.get_fnames(modelfname)
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

    def get_fnames(self, modelfname):
        fnames = glob.glob(op.join(self.output_directory,
                                 modelfname + '_mesh', 'mesh*'))
        if len(fnames) == 0:
            raise ValueError('No density output found for {}'.format(modelfname))

        fnames = sorted(fnames, key=get_density_time)
        times = [get_density_time(f) for f in fnames]
        return fnames, times

    def make_projection_file(self, modelfname, vn, wn):
        projection_exe = op.join(self.MIIND_APPS, 'Projection', 'Projection')
        out = subprocess.check_output(
          [projection_exe, modelfname], cwd=self.xml_location)
        vmax, wmax = np.ceil(np.array(out.split('\n')[3].split(' ')[2:],
                                    dtype=float)).astype(int)
        vmin, wmin = np.floor(np.array(out.split('\n')[4].split(' ')[2:],
                                    dtype=float)).astype(int)
        cmd = [projection_exe, modelfname, vmin, vmax,
             vn, wmin, wmax, wn]
        subprocess.call([str(c) for c in cmd], cwd=self.xml_location)

    def read_projection(self, projfname, vn, wn):
        proj_pathname = op.join(self.xml_location, projfname)
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
        import matplotlib.pyplot as plt
        data_ = self.get_marginal_densities(**args)
        for modelfname, data in data_.items():
            path = op.join(self.output_directory,
                          modelfname.replace('.model', '') +
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

    def plot_density(self, modelfame, densityfname=None,
                     time=None, timestep=None, colorbar=None):
        import visualize
        path = op.join(self.output_directory, modelfame.replace('.model', '') +
                     '_density')
        if not op.exists(path):
            os.mkdir(path)
        fnames, times, idxs = self.get_density_fnames(
            modelfname, time=time, timestep=timestep,
            densityfname=densityfname)
        m = visualize.ModelVisualizer(modelname)
        for fpath, time, idx in zip(fnames, times, idxs):
            fname = op.split(fpath)[-1]
            m.showfile(fpath,
                       pdfname=op.join(path, '%i_'%idx + fname),
                       runningtext='t = %f'%time,
                       colorlegend=colorbar)

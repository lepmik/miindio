import os.path as op
from tools import *
import mesh as meshmod
import glob

class Density:
    def __init__(self, xml_location, output_directory, parameters):
        self.xml_location = xml_location
        self.output_directory = output_directory
        simpar = parameters['Simulation']
        self.modelfiles = [m.get('modelfile')
                           for m in simpar['Algorithms']['Algorithm']
                           if m.get('modelfile') is not None]
        simio = simpar['SimulationIO']
        self.WITH_STATE = simio['WithState']['content']
        if not self.WITH_STATE:
            raise ValueError('State is not saved.')

    def get_marginal_densities(self, modelpath=None, densityfname=None,
                             time=None, timestep=None, vn=100, wn=100,
                             force=False):
        data = {}
        modelfiles = [modelpath] if modelpath is not None else self.modelfiles
        for modelfname in modelfiles:
          projfname = modelfname.replace('.model', '.projection')
          proj, mesh = self.read_projection(projfname, vn, wn)
          fnames, times, idxs = self.get_density_fnames(
              modelfname, time=time, timestep=timestep,
              densityfname=densityfname)

          vs = np.zeros((len(fnames), proj['N_V']))
          ws = np.zeros((len(fnames), proj['N_W']))
          for ii, fname in enumerate(fnames):
              print('Calculating marginals, {} of {}'.format(ii, len(fnames)))
              vs[ii, :], ws[ii, :] = self.get_marginal_density(
                  modelfname, fname, proj, mesh, force)
          bins_v = np.linspace(proj['V_min'], proj['V_max'], proj['N_V'])
          bins_w = np.linspace(proj['W_min'], proj['W_max'], proj['N_W'])
          data[modelfname] = {
              'v': vs, 'w': ws, 'bins_v': bins_v,
              'bins_w': bins_w, 'times': times, 'idxs': idxs}
        # self.set_and_save(fpathout, fname, data)
        return data

    def get_density_fnames(self, modelfname, densityfname=None,
                         time=None, timestep=None):
        fnames = glob.glob(op.join(self.output_directory,
                                 modelfname + '_mesh', 'mesh*'))
        if len(fnames) == 0:
            raise ValueError('No density output found for {}'.format(modelfname))

        fnames = sorted(fnames, key=get_density_time)
        times_ = [get_density_time(f) for f in fnames]
        if timestep is not None:
            assert time is None and densityfname is None
            fnames = fnames[::timestep]
            times = times_[::timestep]
        elif time is not None:
            assert timestep is None and densityfname is None
            if time == 'end':
                time = times_[-1]
            elif not isinstance(time, float):
                raise TypeError('"time" must be a float or the string "end"')
            fnames = [fnames[times_.index(time)]]
            times = [time]
        elif densityfname is not None:
            assert time is None and timestep is None
            fname = [densityfname]
            times = [times[fnames.index(densityfname)]]
        else:
            times = times_
        idxs = [times_.index(time) for time in times]
        return fnames, times, idxs

    def get_marginal_density(self, modelfname, fname, proj,
                             mesh, force=False):
        fpathout = op.join(self.output_directory, 'marginal_density.npz')
        modelfname = op.split(modelfname)[-1]

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

        v = np.zeros(proj['N_V'])
        w = np.zeros(proj['N_W'])
        density, coords = read_density(fname)
        masses = calc_mass(mesh, density, coords)
        for trans in proj['transitions']:
            i, j = [int(a) for a in trans['coordinates'].split(',')]
            cell_mass = masses[coords.index((i, j))]
            if cell_mass < 1e-15:
                continue

            for jj, dd in get_scaling(trans['vbins']):
                v[jj] += cell_mass * dd
            for jj, dd in get_scaling(trans['wbins']):
                w[jj] += cell_mass * dd
        dv = abs(proj['V_max'] - proj['V_min']) / float(proj['N_V'])
        dw = abs(proj['W_max'] - proj['W_min']) / float(proj['N_W'])
        for db, n in zip([dv, dw], [v, w]):
            n = n / db / n.sum()
        return v, w

    def get_or_load(self, fname, name):
        if hasattr(self, name):
            return getattr(self, name)
        if op.exists(fname):
            return np.load(fname)['marginal'][()][name]
        return None

    def set_and_save(self, fname, name, data):
        setattr(self, name, data)
        if op.exists(fname):
            other = np.load(fname)['marginal'][()]
            data = other.update({name: data})
        np.savez(fname, marginal=data)

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
                                  '{}_'.format(data['idxs'][ii]) +
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

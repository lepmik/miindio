import click
from miindio import MiindIO


@click.group()
def cli():
    pass


@cli.command("generate", short_help='Generate a MIIND executable based on a ' +
             '"xml" parameter file.')
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--directory", "-d", type=click.Path(exists=True))
@click.option("--run", "-r", is_flag=True)
def generate_(xml_path, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    io.generate()
    if kwargs['run']:
        io.run()


@cli.command("run", short_help='Run a MIIND executable based on a ' +
             '"xml" parameter file.')
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--directory", "-d", type=click.Path(exists=True))
@click.option("--generate", "-g", is_flag=True)
def run_(xml_path, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    if kwargs['generate']:
        io.generate()
    io.run()


@cli.command("plot-marginal-density",
             short_help='Plot marginal density of a model')
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--model_path", type=click.Path(exists=True))
@click.option("--density_path", type=click.Path(exists=True))
@click.option("--directory", "-d", type=click.Path(exists=True))
@click.option("--n_bins_w", "-w", default=100, type=click.INT)
@click.option("--n_bins_v", "-v", default=100, type=click.INT)
@click.option("--time", "-t", type=click.STRING)
@click.option("--timestep", "-s", type=click.INT)
def plot_marginal_density_(xml_path, **kwargs):
    if kwargs['time'] is not None:
        if kwargs['time'] != 'end':
            kwargs['time'] = float(kwargs['time'])
    io = MiindIO(xml_path, kwargs['directory'])
    io.density.plot_marginal_density(modelpath=kwargs['model_path'],
                             densityfname=kwargs['density_path'],
                             time=kwargs['time'], timestep=kwargs['timestep'],
                             vn=kwargs['n_bins_v'], wn=kwargs['n_bins_w'])


@cli.command("plot-density",
             short_help='Plot 2D density of a model')
@click.argument("xml_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--directory", "-d", type=click.Path(exists=True))
def plot_density_(xml_path, model_path, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    io.density.plot_density(model_path.replace('.model', ''))


def main():
    cli()

if __name__ == "__main__":
    sys.exit(main())

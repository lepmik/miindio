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
@click.argument("model_name", type=click.STRING)
@click.option("--directory", "-d", type=click.Path(exists=True))
@click.option("--n_bins_w", "-w", default=100, type=click.INT)
@click.option("--n_bins_v", "-v", default=100, type=click.INT)
def plot_marginal_density_(xml_path, model_name, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    marginal = io.marginal[model_name]
    marginal.vn = kwargs['n_bins_v']
    marginal.wn = kwargs['n_bins_w']
    marginal.plot()


@cli.command("plot-density",
             short_help='Plot 2D density of a model')
@click.argument("xml_path", type=click.Path(exists=True))
@click.argument("model_name", type=click.STRING)
@click.option("--directory", "-d", type=click.Path(exists=True))
def plot_density_(xml_path, model_name, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    density = io.density[model_name]
    for fname in density.fnames:
        density.plot_density(fname)


def main():
    cli()

if __name__ == "__main__":
    sys.exit(main())

import click
from miindio import MiindIO


@click.group()
def cli():
    pass


@cli.command("generate", short_help='Generate a MIIND executable based on a ' +
             '"xml" parameter file.')
@click.argument("xml_path", type=click.Path(exists=True))
@click.option("--directory", "-d", type=click.Path(exists=True))
def generate_(name, xml_path, **kwargs):
    io = MiindIO(xml_path, kwargs['directory'])
    io.generate()


def main():
    cli()

if __name__ == "__main__":
    sys.exit(main())

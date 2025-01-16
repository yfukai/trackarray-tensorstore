"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Tensorstore Trackarr."""


if __name__ == "__main__":
    main(prog_name="tensorstore-trackarr")  # pragma: no cover

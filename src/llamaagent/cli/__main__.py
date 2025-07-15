"""Allow `python -m llamaagent.cli …` and `python -m llamaagent …`."""

from llamaagent.cli import cli

if __name__ == "__main__":  # pragma: no cover
    cli()  # pylint: disable=no-value-for-parameter

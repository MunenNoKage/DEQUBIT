"""Seeding, logging setup and environment capture."""


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch random generators."""
    raise NotImplementedError  # TODO


def setup_logging(level: str) -> None:
    """Configure the root logger with the level from the config."""
    raise NotImplementedError  # TODO


def collect_environment() -> dict[str, str]:
    """Python and library versions, OS and device used for the run."""
    raise NotImplementedError  # TODO

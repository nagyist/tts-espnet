"""CLI utils methods."""

import sys
from collections.abc import Sequence
from distutils.util import strtobool as dist_strtobool

import numpy


def strtobool(x):
    """Convert string to boolean."""
    # distutils.util.strtobool returns integer, but it's confusing,
    return bool(dist_strtobool(x))


def get_commandline_args():
    """Get command line arguments."""
    extra_chars = [
        " ",
        ";",
        "&",
        "(",
        ")",
        "|",
        "^",
        "<",
        ">",
        "?",
        "*",
        "[",
        "]",
        "$",
        "`",
        '"',
        "\\",
        "!",
        "{",
        "}",
    ]

    # Escape the extra characters for shell
    argv = [
        (
            arg.replace("'", "'\\''")
            if all(char not in arg for char in extra_chars)
            else "'" + arg.replace("'", "'\\''") + "'"
        )
        for arg in sys.argv
    ]

    return sys.executable + " " + " ".join(argv)


def is_scipy_wav_style(value):
    """Check if value is a tuple or not."""
    # If Tuple[int, numpy.ndarray] or not
    return (
        isinstance(value, Sequence)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], numpy.ndarray)
    )


def assert_scipy_wav_style(value):
    """Assert if value is in scipy wav style."""
    assert is_scipy_wav_style(
        value
    ), "Must be Tuple[int, numpy.ndarray], but got {}".format(
        type(value)
        if not isinstance(value, Sequence)
        else "{}[{}]".format(type(value), ", ".join(str(type(v)) for v in value))
    )

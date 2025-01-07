import os
import shutil
import tempfile
import unittest
from pathlib import Path


def str_to_bool(value) -> int:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {value}")


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def parse_flag_from_env(key, default=False):
    """Returns truthy value for `key` from the env if available else the default."""
    value = os.environ.get(key, str(default))
    return (
        str_to_bool(value) == 1
    )  # As its name indicates `str_to_bool` actually returns an int...


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)


def skip(test_case):
    "Decorator that skips a test unconditionally"
    return unittest.skip("Test was skipped")(test_case)


def slow(test_case):
    """
    Decorator marking a test as slow. Slow tests are skipped by default. Set the RUN_SLOW environment variable to a
    truthy value to run them.
    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


class TempDirTestCase(unittest.TestCase):
    """
    A TestCase class that keeps a single `tempfile.TemporaryDirectory` open for the duration of the class, wipes its
    data at the start of a test, and then destroyes it at the end of the TestCase.

    Useful for when a class or API requires a single constant folder throughout it's use, such as Weights and Biases

    The temporary directory location will be stored in `self.tmpdir`
    """

    clear_on_setup = True

    @classmethod
    def setUpClass(cls):
        "Creates a `tempfile.TemporaryDirectory` and stores it in `cls.tmpdir`"
        cls.tmpdir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        "Remove `cls.tmpdir` after test suite has finished"
        if os.path.exists(cls.tmpdir):
            shutil.rmtree(cls.tmpdir)

    def setUp(self):
        "Destroy all contents in `self.tmpdir`, but not `self.tmpdir`"
        if self.clear_on_setup:
            for path in self.tmpdir.glob("**/*"):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)

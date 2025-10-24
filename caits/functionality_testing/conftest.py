from _pytest.terminal import TerminalReporter
from colorama import init
from caits.functionality_testing import config
import numpy as np
import pytest

init(autoreset=True)
def pytest_terminal_summary(terminalreporter: TerminalReporter):
    terminalreporter.section("Functionality Test", blue=True, bold=True)

    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    error = len(terminalreporter.stats.get('error', []))
    skipped = len(terminalreporter.stats.get('skipped', []))

    terminalreporter.write_line(config.SUCCESS_COLOR + f"✔ Passed: {passed}" + config.RESET)
    terminalreporter.write_line(config.FAIL_COLOR + f"✘ Failed: {failed}" + config.RESET)
    terminalreporter.write_line(config.ERROR_COLOR + f"! Error: {error}" + config.RESET)
    terminalreporter.write_line(config.SKIP_COLOR + f"➜ Skipped: {skipped}" + config.RESET)


params = [
    ((1000,), 0),
    ((1000, 1), 0),
    ((1000, 2), 0),
    ((1, 1000), 1),
    ((2, 1000), 1),
]

@pytest.fixture(params=params, ids=lambda arr: f"shape={arr[0]}-axis={arr[1]}")
def arr(request):
    return np.random.rand(*request.param[0]), request.param[1]

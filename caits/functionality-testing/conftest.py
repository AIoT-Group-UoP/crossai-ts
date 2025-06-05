from _pytest.reports import TestReport
from _pytest.terminal import TerminalReporter
from _pytest._code.code import ReprEntry, ExceptionChainRepr, ReprTraceback
from colorama import init

import config

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
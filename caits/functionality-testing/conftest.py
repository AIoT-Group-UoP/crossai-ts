from colorama import init
import config

init(autoreset=True)

def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(config.INFO_COLOR + "\nFunctionality Test Summary:" + config.RESET)

    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))

    terminalreporter.write_line(config.SUCCESS_COLOR + f"✔ Passed: {passed}" + config.RESET)
    terminalreporter.write_line(config.FAIL_COLOR + f"✘ Failed: {failed}" + config.RESET)
    terminalreporter.write_line(config.SKIP_COLOR + f"➜ Skipped: {skipped}" + config.RESET)



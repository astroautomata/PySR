"""Marker registration for the documentation-example suite.

The slow examples are real searches measured in minutes, so they are opt-in. A bare
`pytest` run exercises every example that finishes in about a minute.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run the examples whose searches take minutes",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: a search measured in minutes")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    import pytest

    skip = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

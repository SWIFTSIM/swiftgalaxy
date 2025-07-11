Contributing to SWIFTGalaxy
===========================

Contributions for SWIFTGalaxy should come as pull requests submitted to our [GitHub repository](https://github.com/SWIFTSIM/swiftgalaxy).

Contributions are always welcome, but you should make sure of the following:

+ Your contributions include type hints (in function and method declarations).
+ Your contributions are formatted with the `black` formatter.
+ Your contributions do not have formatting or style issues identified by `flake8`.
+ Your contributions are covered by tests and pass all unit tests in `/tests`.
+ Your contributions add unit tests for new functionality.
+ Your contributions are documented fully under `/docs`.

Some brief quickstart-style notes are included below, but are not intended to replace consulting the documentation of each relevant toolset. We recognize that this can seem daunting to users new to collaborative development. Don't hesitate to get in touch for help if you want to contribute!

Black style
-----------

You can install the `black` formatter with `pip install black`. To check your copy of the repository you can then run `black --check --diff` in the same directory as the `pyproject.toml` file. A message like `All done! ✨ 🍰 ✨` indicates that your working copy passes the checks, while `Oh no! 💥 💔 💥` indicates problems are present. You can also use `black` to automatically edit your copy of the repository to comply with the style rules by running `black` in the same directory as `pyproject.toml`. Don't forget to commit any changes it makes. In most cases, code formatted with `black` will pass the `flake8` checks.

Flake8 style guide enforcement
------------------------------

You can install the `flake8` style guide compliance checker with `pip install flake8`. Contributions will be checked on github on `python3.12`. To check your copy of the repository for compliance with the [PEP8](https://peps.python.org/pep-0008/) style guide, run `flake8` in the same directory as the `pyproject.toml` file. No message indicates success (if unsure, you can check the return code with `flake8;echo $?`, in this case a value of 0 indicates success). In case of failure, any errors and warnings will be printed. In general these should be addressed by editing the code to be compliant (running `black` will resolve most issues automatically). In some rare cases it may be appropriate to ignore an error by editing the `/.flake8` file. Note that the maximum line length for the project is set to 90 characters.

Pytest unit testing
-------------------

You can install the `pytest` unit testing toolkit with `pip install pytest`. You can then run `pytest` in the same directory as the `pyproject.toml` file to run the existing unit tests. Any test failures will report detailed debugging information. Note that the tests on github are run with python versions `3.10`, `3.11`, `3.12` and `3.13`, and the latest PyPI releases of the relevant dependencies (swiftsimio, unyt, etc.). To run only tests in a specific file, you can do e.g. `pytest tests/test_creation.py`. The tests to be run can be further narrowed down with the `-k` argument to `pytest` (see `pytest --help`).

Test coverage is provided by [Codecov](https://about.codecov.io/). This tool checks which lines of code are run during the execution of the test suite. While having all code lines run (called "100% test coverage"), or nearly, does not guarantee good tests, it is still a useful benchmark. Pull requests opened on SWIFTGalaxy will produce a test coverage report once the test suite finishes running. This will flag any new code lines not covered by tests, or previously covered lines that are no longer covered, etc. You should look at this and evaluate whether it reveals additional useful test cases that you hadn't considered. You can also generate a code coverage report locally with the `pytest-cov` extension for `pytest`.

Documentation
-------------

The API documentation is built automatically from the docstrings of classes, functions, etc. in the source files. These follow the NumPy-style format. All public (i.e. not starting in `_`) modules, functions, classes, methods, etc. should have an appropriate docstring. In addition to this there is "narrative documentation" that should describe the features of the code. The docs are built with `sphinx` and use the "ReadTheDocs" theme. If you have the dependencies installed (check `/docs/requirements.txt`) you can build the documentation locally with `make html` in the `/docs` directory. Opening the `/docs/index.html` file with a browser will then allow you to browse the documentation and check your contributions.

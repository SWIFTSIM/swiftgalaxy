Getting Started
===============

:mod:`swiftgalaxy` facilitates analyses of individual galaxies from cosmological hydrodynamical simulations executed with the `SWIFT`_ code. The core :class:`~swiftgalaxy.reader.SWIFTGalaxy` class makes selecting particles belonging to a galaxy, transforming their coordinates, masking its particle arrays, working with spherical or cylindrical coordinates, accessing its integrated properties computed by a halo finder, and more, easy. It is built upon the :class:`~swiftsimio.reader.SWIFTDataset` from :mod:`swiftsimio` and takes advantage of that module's lazy-loading and unit-awareness features. Users already familiar with usage of :mod:`swiftsimio` will find that any syntax to interact with a :class:`~swiftsimio.reader.SWIFTDataset` also works with a :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

.. _SWIFT: http://swift.dur.ac.uk

Requirements
------------

``python3.10`` or higher is required.

Python packages
^^^^^^^^^^^^^^^

Required:

+ :mod:`swiftsimio` (v10 or higher), required to provide the :class:`~swiftsimio.reader.SWIFTDataset` class and related functionality (note that :mod:`swiftsimio` has additional required dependencies).
+ :mod:`numpy`, required to support various array operations.
+ :mod:`unyt`, required for unit calculations.
+ :mod:`scipy`, required to specify rotations via the :class:`~scipy.spatial.transform.Rotation` class.

Optional:

+ :mod:`velociraptor`, required to enable support for Velociraptor_ halo finder outputs.
+ :mod:`soap` (`soap on github`_ not the :mod:`soap` from PyPI!), for generating example SOAP_ data sets.
+ :mod:`caesar` (`caesar on github`_ not the :mod:`caesar` from PyPI!), required to enable support for Caesar_ catalogues.
+ :mod:`astropy`, used in generating example data sets.
+ :mod:`h5py`, used to generate example data sets.

.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/abstract
.. _soap on github: https://github.com/SWIFTSIM/SOAP
.. _SOAP: https://swiftsimio.readthedocs.io/en/latest/soap/index.html
.. _caesar on github: https://github.com/dnarayanan/caesar
.. _Caesar: https://caesar.readthedocs.io/en/latest/

Additional optional packages for developers:

+ :mod:`ruff` for code formatting and style checks.
+ :mod:`mypy` for type checking (run ``mypy --install-types --non-interactive`` after installation).
+ :mod:`pytest` to run the test suite.
+ :mod:`numpydoc` to check for issues in docstrings.
+ :mod:`flake8` to check for code style issues.
+ :mod:`pytest-cov` to generate test coverage reports

And for building the docs:

+ :mod:`sphinx`
+ :mod:`sphinx-rtd-theme`
+ :mod:`sphinx-design`
+ :mod:`recommonmark`


Installing
----------

:mod:`swiftgalaxy` can be installed using the python packaging manager, ``pip``, or any other packaging manager that you wish to use:

``pip install swiftgalaxy``

Note that this will also install required dependencies. To install optional dependencies to support Velociraptor catalogues use ``pip install swiftgalaxy[velociraptor]``. For SOAP and Caesar support use:

.. code-block::

   pip install git+https://github.com/SWIFTSIM/soap.git@soap_runtime
   pip install git+https://github.com/dnarayanan/caesar.git

The SOAP package is only needed if you want to generate example input files for SOAP, otherwise there are no additional dependencies.

Installation for development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To set up the code for development, first clone the latest `main` branch from `github`_:
``git clone https://github.com/SWIFTSIM/swiftgalaxy.git``
and install with ``pip`` using the ``-e`` (editable) flag, and specify development dependencies with ``[dev]``:
``pip install -e swiftgalaxy/[dev]``.

You should also install type definitions for :mod:`mypy`:
``mypy --install-types --non-interactive``.

You can check that the installation and your environment is ready for development work by moving to the code root directory (``cd swiftgalaxy``) and running the following checks:

.. code-block:: bash

   flake8
   ruff format --check
   ruff check
   mypy
   python -m numpydoc lint swiftgalaxy/*.py
   pytest --cov --cov-branch

You may wish to set up the following `pre-commit hook`_:

.. code-block:: bash

   #!/bin/sh

   flake8 || exit 1
   ruff format --check || exit 1
   ruff check || exit 1
   mypy || exit 1
   python -m numpydoc lint swiftgalaxy**/*.py || exit 1

   exit 0

and `pre-push hook`_:

.. code-block:: bash

   #!/bin/sh

   remote="$1"
   url="$2"

   flake8 || exit 1
   ruff format --check || exit 1
   ruff check || exit 1
   mypy || exit 1
   python -m numpydoc lint swiftgalaxy**/*.py || exit 1
   pytest || exit 1

   exit 0

.. _github: https://github.com/SWIFTSIM/swiftgalaxy
.. _pre-commit hook: https://git-scm.com/book/ms/v2/Customizing-Git-Git-Hooks
.. _pre-push hook: https://git-scm.com/book/ms/v2/Customizing-Git-Git-Hooks

.. _quick-start:

Quick start
-----------

.. include:: ../../../README.rst
   :start-after: EXAMPLE_START_LABEL
   :end-before: EXAMPLE_END_LABEL

Like a :class:`~swiftsimio.reader.SWIFTDataset`, all data are loaded 'lazily' on demand.

However, information from the halo catalogue is used to select only the particles identified as bound to this galaxy. The coordinate system is centred in both position and velocity on the centre and peculiar velocity of the galaxy, as determined by the halo catalogue. The coordinate system can be further manipulated, and all particle arrays will stay in a consistent reference frame at all times.

Again like for a :class:`~swiftsimio.reader.SWIFTDataset`, the units and metadata information are available:

.. code-block:: python

    sg.units
    sg.metadata

:mod:`swiftgalaxy` supports Python's tab completion features. This means that you can browse the available attributes of objects in an interactive interpreter by starting to type an attribute (or just a trailing dot) and pressing tab twice. A few examples to help start exploring:

   - ``sg.<tab><tab>``
   - ``sg.gas.<tab><tab>``
   - ``sg.halo_catalogue.<tab><tab>``

The further features of a :class:`~swiftgalaxy.reader.SWIFTGalaxy` are detailed in the next sections.

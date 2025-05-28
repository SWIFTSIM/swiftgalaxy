SWIFTGalaxy
===========

|Python version| |PyPI version| |ASCL| |Repostatus| |Zenodo| |Build status| |Documentation status| |codecov|


.. |Build status| image:: https://github.com/SWIFTSIM/swiftgalaxy/actions/workflows/lint_and_test.yml/badge.svg
    :target: https://github.com/SWIFTSIM/swiftgalaxy/actions/workflows/lint_and_test.yml
    :alt: Build Status
.. |Documentation status| image:: https://readthedocs.org/projects/swiftgalaxy/badge/?version=latest
    :target: https://swiftgalaxy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |Python version| image:: https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSWIFTSIM%2Fswiftgalaxy%2Fmain%2Fpyproject.toml
   :alt: Python Version from PEP 621 TOML
.. |PyPI version| image:: https://img.shields.io/pypi/v/swiftgalaxy
   :target: https://pypi.org/project/swiftgalaxy/
   :alt: PyPI - Version
.. |Repostatus| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active
.. |codecov| image:: https://codecov.io/gh/SWIFTSIM/swiftgalaxy/graph/badge.svg?token=YV3YYEK78Z 
   :target: https://codecov.io/gh/SWIFTSIM/swiftgalaxy
.. |Zenodo| image:: https://zenodo.org/badge/488271795.svg
   :target: https://doi.org/10.5281/zenodo.15502355
.. |ASCL| image:: https://img.shields.io/badge/ascl-2505.002-blue.svg?colorB=262255
   :alt: ascl:2505.002
   :target: https://ascl.net/2505.002

.. INTRO_START_LABEL

SWIFTGalaxy is a module that extends SWIFTSimIO_ tailored to analyses of particles belonging to individual simulated galaxies. It inherits from and extends the functionality of the ``SWIFTDataset``. It understands the content of halo catalogues (supported: `Velociraptor`_, `Caesar`_, `SOAP`_) and therefore which particles belong to a galaxy or other group of particles, and its integrated properties. The particles occupy a coordinate frame that is enforced to be consistent, such that particles loaded on-the-fly will match e.g. rotations and translations of particles already in memory. Intuitive masking of particle datasets is also enabled. Utilities to make working in cylindrical and spherical coordinate systems more convenient are also provided. Finally, tools to iterate efficiently over multiple galaxies are also provided.

Installation is as simple as ``pip install swiftgalaxy``. ``numpy>=2.0`` is required and will be installed automatically as a dependency.

.. _SWIFTSimIO: http://swiftsimio.readthedocs.org
.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/abstract
.. _Caesar: https://caesar.readthedocs.io/en/latest/
.. _SOAP: https://github.com/SWIFTSIM/SOAP
.. _PyPI: https://pypi.org

.. INTRO_END_LABEL

+ `Quick start guide`_
+ `Full documentation`_

.. _Quick start guide: https://swiftgalaxy.readthedocs.io/en/latest/getting_started
.. _Full documentation: https://swiftgalaxy.readthedocs.io/en/latest
   
Citing SWIFTGalaxy
------------------

.. CITING_START_LABEL

If your use of SWIFTGalaxy leads to a publication, please cite the `swiftgalaxy entry`_ in the ASCL_ (`indexed on ADS`_). Ideally specify the version used (`Zenodo DOI`_, git commit ID and/or version number) and link to the github repository. A PyOpenSci_ review leading to a JOSS_ submission is `in progress`_, please check back for updated citation information.

.. code-block:: bibtex

    @MISC{
        2025ascl.soft05002O,
     	author = {{Oman}, Kyle A.},
        title = "{swiftgalaxy}",
        keywords = {Software},
        howpublished = {Astrophysics Source Code Library, record ascl:2505.002},
        year = 2025,
        month = may,
        eid = {ascl:2505.002},
        pages = {ascl:2505.002},
        archivePrefix = {ascl},
        eprint = {2505.002},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2025ascl.soft05002O},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Please also consider the `citations requested for SWIFTSimIO <citeSWIFTSimIO>`_.

.. _swiftgalaxy entry: https://ascl.net/2505.002
.. _ASCL: https://ascl.net
.. _JOSS: https://joss.theoj.org/
.. _PyOpenSci: https://www.pyopensci.org/
.. _in progress: https://github.com/pyOpenSci/software-submission/issues/244#issuecomment-2912506097
.. _indexed on ADS: https://ui.adsabs.harvard.edu/abs/2025ascl.soft05002O
.. _Zenodo DOI: https://zenodo.org/records/15502385
.. _citeSWIFTSimIO: https://swiftsimio.readthedocs.io/en/latest/index.html#citing-swiftsimio

.. CITING_END_LABEL

Community
---------

Code contributions are very welcome! A good place to start is the [contributing guide](https://github.com/SWIFTSIM/swiftgalaxy/blob/main/CONTRIBUTING.md) and how to set up a [development environment](https://swiftgalaxy.readthedocs.io/en/latest/getting_started/index.html#installing).

SWIFTGalaxy is licensed under [GPL-3.0](https://github.com/SWIFTSIM/swiftgalaxy/tree/main?tab=GPL-3.0-1-ov-file) and community members are expected to abide by the (code of conduct)[https://github.com/SWIFTSIM/swiftgalaxy/tree/main?tab=coc-ov-file].

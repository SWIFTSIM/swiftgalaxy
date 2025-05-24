SWIFTGalaxy
===========

|Python version| |PyPI version| |Repostatus| |Zenodo| |Build status| |Documentation status| |codecov|


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

.. INTRO_START_LABEL

SWIFTGalaxy is a module that extends SWIFTSimIO_ tailored to analyses of particles belonging to individual simulated galaxies. It inherits from and extends the functionality of the ``SWIFTDataset``. It understands the content of halo catalogues (supported: `Velociraptor`_, `Caesar`_, `SOAP`_) and therefore which particles belong to a galaxy or other group of particles, and its integrated properties. The particles occupy a coordinate frame that is enforced to be consistent, such that particles loaded on-the-fly will match e.g. rotations and translations of particles already in memory. Intuitive masking of particle datasets is also enabled. Finally, some utilities to make working in cylindrical and spherical coordinate systems more convenient are also provided.

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

.. note::

   ``swiftgalaxy`` will be listed on the ASCL_ after the first publication using it. Citation details will then be given here. In the meantime, if you publish a paper using ``swiftgalaxy``, please get in touch at submission time!

Please also consider the `citations requested for SWIFTSimIO <citeSWIFTSimIO>`_.

.. _ASCL: https://ascl.net
.. _indexed on ADS: https://ui.adsabs.harvard.edu/abs/20??ascl.soft?????O
.. _citeSWIFTSimIO: https://swiftsimio.readthedocs.io/en/latest/index.html#citing-swiftsimio

.. CITING_END_LABEL

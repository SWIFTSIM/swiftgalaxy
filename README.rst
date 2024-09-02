SWIFTGalaxy
===========

|Build status| |Documentation status|

.. |Build status| image:: https://github.com/SWIFTSIM/swiftgalaxy/actions/workflows/code_quality.yml/badge.svg
    :target: https://github.com/SWIFTSIM/swiftgalaxy/actions/workflows/black_pytest.yml
    :alt: Build Status
.. |Documentation status| image:: https://readthedocs.org/projects/swiftgalaxy/badge/?version=latest
    :target: https://swiftgalaxy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. INTRO_START_LABEL

SWIFTGalaxy is a module that extends SWIFTSimIO_ tailored to analyses of particles belonging to individual simulated galaxies. It inherits from and extends the functionality of the ``SWIFTDataset``. It understands the content of halo catalogues (supported: `Velociraptor`_, `Caesar`_, `SOAP`_) and therefore which particles belong to a galaxy or other group of particles, and its integrated properties. The particles occupy a coordinate frame that is enforced to be consistent, such that particles loaded on-the-fly will match e.g. rotations and translations of particles already in memory. Intuitive masking of particle datasets is also enabled. Finally, some utilities to make working in cylindrical and spherical coordinate systems more convenient are also provided.

.. warning::

   In order to support use of SOAP halo catalogues this version (1.1.0) of SWIFTGalaxy depends on the `load_fof_catalogues`_ branch of SWIFTSimIO. This dependency will be automatically handled by ``pip``, but to use SOAP catalogues in SWIFTGalaxy you **must** install from github. One way to do this is ``pip install git+https://github.com/SWIFTSIM/swiftgalaxy.git``. PyPI_ does not allow depending on github repositories so SWIFTGalaxy v1.1.0 will not be released on PyPI (and therefore cannot be installed with ``pip install swiftgalaxy``) until the load_fof_catalogues branch is merged into SWIFTSimIO and itself released on PyPI. You can use the PyPI versions of both SWIFTGalaxy and SWIFTSimIO (these are compatible) but in this case SOAP halo catalogues are not supported.

.. _SWIFTSimIO: http://swiftsimio.readthedocs.org
.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/abstract
.. _Caesar: https://caesar.readthedocs.io/en/latest/
.. _SOAP: https://github.com/SWIFTSIM/SOAP
.. _load_fof_catalogues: https://github.com/SWIFTSIM/swiftsimio/tree/load_fof_catalogues
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

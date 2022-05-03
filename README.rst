SWIFTGalaxy
===========

|Build status| |Documentation status|

.. |Build status| image:: https://github.com/SWIFTSIM/swiftgalaxy/actions/workflows/pytest.yml/badge.svg
    :target: https://github.com/SWIFTSIM/swiftgalaxy/actions/workflows/pytest.yml
    :alt: Build Status
.. |Documentation status| image:: https://readthedocs.org/projects/swiftgalaxy/badge/?version=latest
    :target: https://swiftgalaxy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

+ `Quick start guide`_
+ `Full documentation`_

.. _Quick start guide: https://kyleaoman.github.io/swiftgalaxy/build/html/getting_started
.. _Full documentation: https://kyleaoman.github.io/swiftgalaxy 

rtd_start_here

SWIFTGalaxy is a module that extends SWIFTSimIO_ tailored to analyses of particles belonging to individual simulated galaxies. It inherits from and extends the functionality of the ``SWIFTDataset``. It understands the output of a halo finder (supported: `Velociraptor`_; planned support: `HBT+`_) and therefore which particles belong to a galaxy, and its integrated properties. The particles occupy a coordinate frame that is enforced to be consistent, such that particles loaded on-the-fly will match e.g. rotations and translations of particles already in memory. Intuitive masking of particle datasets is also enabled. Finally, some utilities to make working in cylindrical and spherical coordinate systems more convenient are also provided.

.. _SWIFTSimIO: http://swiftsimio.readthedocs.org
.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/abstract
.. _HBT+: https://ui.adsabs.harvard.edu/abs/2018MNRAS.474..604H/abstract

Citing SWIFTGalaxy
------------------

.. note::

   These entries have not gone live yet.

Please cite the `swiftgalaxy entry`_ in the `ASCL`_ (`indexed on ADS`_):

.. code-block:: bibtex

    @MISC{20??ascl.soft?????O,
           author = {{Oman}, Kyle A.},
            title = "{SWIFTGalaxy}",
         keywords = {Software},
     howpublished = {Astrophysics Source Code Library, record ascl:????.???},
             year = 20??,
            month = ???,
              eid = {ascl:????.???},
            pages = {ascl:????.???},
    archivePrefix = {ascl},
           eprint = {????.???},
           adsurl = {https://ui.adsabs.harvard.edu/abs/20??ascl.soft?????O},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Please also consider the `citations requested for SWIFTSimIO <citeSWIFTSimIO>`_.

.. _swiftgalaxy entry: https://ascl.net/????.???
.. _ASCL: https://ascl.net
.. _indexed on ADS: https://ui.adsabs.harvard.edu/abs/20??ascl.soft?????O
.. _citeSWIFTSimIO: https://swiftsimio.readthedocs.io/en/latest/index.html#citing-swiftsimio

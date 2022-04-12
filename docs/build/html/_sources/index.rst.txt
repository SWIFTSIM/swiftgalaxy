SWIFTGalaxy documentation
=========================

:mod:`swiftgalaxy` is a module that extends :mod:`swiftsimio` tailored to analyses of particles belonging to individual simulated galaxies. The main class, :class:`~swiftgalaxy.reader.SWIFTGalaxy`, inherits from and extends the functionality of the :class:`~swiftsimio.reader.SWIFTDataset`. It understands the output of a halo finder (supported: `Velociraptor`_; planned support: `HBT+`_) and therefore which particles belong to a galaxy, and its integrated properties. The particles occupy a coordinate frame that is enforced to be consistent, such that particles loaded on-the-fly will match e.g. rotations and translations of particles already in memory. Intuitive masking of particle datasets is also enabled. Finally, some utilities to make working in cylindrical and spherical coordinate systems more convenient are also provided.

.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/abstract
.. _HBT+: https://ui.adsabs.harvard.edu/abs/2018MNRAS.474..604H/abstract

.. toctree::
   :maxdepth: 2

   getting_started/index
   coordinate_transformations/index
   additional_coordinates/index
   halo_finders/index
   masking/index
   modules/index

Citing SWIFTGalaxy
==================

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

.. _swiftgalaxy entry: https://ascl.net/????.???
.. _ASCL: https://ascl.net
.. _indexed on ADS: https://ui.adsabs.harvard.edu/abs/20??ascl.soft?????O


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

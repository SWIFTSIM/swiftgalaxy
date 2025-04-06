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

+ :mod:`swiftsimio`, required to provide the :class:`~swiftsimio.reader.SWIFTDataset` class and related functionality (note that :mod:`swiftsimio` has additional required dependencies).
+ :mod:`numpy`, required to support various array operations.
+ :mod:`unyt`, required for unit calculations.

Optional:

+ :mod:`scipy`, required to specify rotations via the :class:`~scipy.spatial.transform.Rotation` class.
+ :mod:`velociraptor`, required to enable support for `Velociraptor`_ halo finder outputs.

.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/abstract


Installing
----------

:mod:`swiftgalaxy` can be installed using the python packaging manager, ``pip``, or any other packaging manager that you wish to use:

``pip install swiftgalaxy``

Note that this will also install required dependencies.

To set up the code for development, first clone the latest master from `github`_:

``git clone https://github.com/SWIFTSIM/swiftgalaxy.git``

and install with ``pip`` using the ``-e`` flag,

``pip install -e swiftgalaxy/``

.. _github: https://github.com/SWIFTSIM/swiftgalaxy


Quick start
-----------

Assuming we have a snapshot file :file:`{snap}.hdf5`, and a halo catalogue provided by Velociraptor :file:`{halos}.properties`, :file:`{halos}.catalog_groups`, etc., with the default names for the arrays of coordinates, velocities and particle_ids, we can initialise a :class:`SWIFTGalaxy` for the first row (indexed from 0) in the halo catalogue very easily:

.. code-block:: python

    from swiftgalaxy import SWIFTGalaxy, Velociraptor
    sg = SWIFTGalaxy(
        'snap.hdf5',
        Velociraptor(
            'halos',
            halo_index=0
        )
    )

Like a :class:`~swiftsimio.reader.SWIFTDataset`, the particle datasets are accessed as below, and all data are loaded 'lazily', on demand.

.. code-block:: python

    sg.gas.particle_ids
    sg.dark_matter.coordinates

However, information from the halo catalogue is used to select only the particles identified as bound to this galaxy. The coordinate system is centred in both position and velocity on the centre and peculiar velocity of the galaxy, as determined by the halo finder. The coordinate system can be further manipulated, and all particle arrays will stay in a consistent reference frame at all times.

Again like for a :class:`~swiftsimio.reader.SWIFTDataset`, the units and metadata are available:

.. code-block:: python

    sg.units
    sg.metadata

The halo catalogue interface is accessible as shown below. What this interface looks like depends on the halo finder being used, but will provide values for the individual galaxy of interest.

.. code-block:: python

    sg.halo_catalogue

In this case with :class:`~swiftgalaxy.halo_catalogues.Velociraptor`, we can get the virial mass like this:

.. code-block:: python

    sg.halo_catalogue.masses.mvir

:mod:`swiftgalaxy` supports Python's tab completion features. This means that you can browse the available attributes of objects in an interactive interpreter by starting to type an attribute (or just a trailing dot) and pressing tab twice. A few examples to help start exploring:

   - ``sg.<tab><tab>``
   - ``sg.gas.<tab><tab>``
   - ``sg.halo_catalogue.<tab><tab>``

The further features of a :class:`~swiftgalaxy.reader.SWIFTGalaxy` are detailed in the next sections.

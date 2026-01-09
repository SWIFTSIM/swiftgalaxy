Iterating SWIFTGalaxies
=======================

When similar operations need to be applied to multiple :class:`~swiftgalaxy.reader.SWIFTGalaxy` objects it often makes sense to iterate over them (possibly in parallel, see below). The :class:`~swiftgalaxy.iterator.SWIFTGalaxies` class aims to make this convenient and efficient, mostly by internally managing the order of iteration to avoid re-reading data from disk as much as possible - most often when two objects of interest share a common top-level cell in a SWIFT simulation.

Setting up a :class:`~swiftgalaxy.iterator.SWIFTGalaxies` instance is deliberately designed to be as similar to creating a :class:`~swiftgalaxy.reader.SWIFTGalaxy` as possible. There is one difference: the halo catalogue helper class is initialized with a list of targets instead of a single target identifier.

The :class:`~swiftgalaxy.iterator.SWIFTGalaxies` class otherwise broadly accepts the same initialization arguments as the :class:`~swiftgalaxy.reader.SWIFTGalaxy` class, such as ``auto_recentre``.

It is crucial to know that the order of iteration used by a :class:`~swiftgalaxy.iterator.SWIFTGalaxies` is internally optimized; see the `order of iteration`_ section below.

Multiple-target halo catalogues
-------------------------------

The halo catalogue helper classes (such as :class:`~swiftgalaxy.halo_catalogues.SOAP`, :class:`~swiftgalaxy.halo_catalogues.Velociraptor`, :class:`~swiftgalaxy.halo_catalogues.Caesar`, ...) can be initialized with a list of targets instead of a single target identifier (:mod:`numpy` arrays or :mod:`unyt` arrays are also OK). This results in some new behaviour:
 - When a halo catalogue is created with multiple targets, accessing attributes of the object (such as via the :class:`~swiftgalaxy.reader.SWIFTGalaxy` ``halo_catalogue`` attribute) will return the relevant property of all targets in the catalogue. For example:

.. code-block:: python

   soap = SOAP("my_soap_catalogue.hdf5", soap_index=[11, 22, 33])
   m200 = soap.spherical_overdensity_200_crit.total_mass
   
sets ``m200`` to something like ``cosmo_array([1.2e+12, 3.4e+12, 5.6e+12], dtype=float32, units='1.98841586e+30*kg', comoving=False)``. Keep in mind that the format of halo catalogue values depends on the halo catalogue, keeping formats that should already be familiar to users of each catalogue type.

Once iteration over the targets contained in a :class:`~swiftgalaxy.iterator.SWIFTGalaxies` begins, this behaviour changes: during an iteration step, the catalogue is "masked" down to a single object and behaves just like it would for a single-target :class:`~swiftgalaxy.reader.SWIFTGalaxy`:

.. code-block:: python

   sgs = SWIFTGalaxies("my_snapshot.hdf5", soap)
   for sg in sgs:
      m200 = sg.halo_catalogue.spherical_overdensity_200_crit.total_mass
      
Then on each iteration the value of ``m200`` will look similar to ``cosmo_array([1.2e+12], dtype=float32, units='1.98841586e+30*kg', comoving=False)``.

Order of iteration
------------------

.. warning::
   The most important thing to remember when using :class:`~swiftgalaxy.iterator.SWIFTGalaxies` is that it determines the best order to iterate over your chosen galaxies itself to minimize I/O operations. Be careful not to assume that the iteration is in the order of the target identifiers that you passed to the halo catalogue helper class (such as :class:`~swiftgalaxy.halo_catalogues.SOAP`).

The main purpose of the :class:`~swiftgalaxy.iterator.SWIFTGalaxies` class is to determine an order to iterate through the list of target objects without duplicating I/O operations more than necessary.

In brief, the class evaluates two iteration schemes at initialization and chooses the one that will be most efficient. The first scheme is the "sparse" solution. This determines which top-level cells need to be read for each target galaxy and groups targets together when they share a common region. The second scheme is the "dense" solution. This determines the largest region needed by any single target and then covers the simulation volume in regions of that size on a grid. Targets are then assigned to the region with the closest centre to collect them into groups. Any regions without targets in them are discarded. The bounding box of each region is then adjusted to the minimum extent needed to contain the targets assigned to it. This result in a final set of (probably overlapping) regions. This second scheme is most often optimal when there are many closely packed targets that often overlap the boundaries of top-level cells, such as when iterating over all galaxies in a simulation volume.

The iteration order of the targets is available from the :attr:`~swiftgalaxy.iterator.SWIFTGalaxies.iteration_order` property. However, if obtaining ordered results is required, using the :func:`~swiftgalaxy.iterator.SWIFTGalaxies.map` method is usually a more convenient approach than e.g. sorting the output of iteration in the optimal order.

Map
...

The :func:`swiftgalaxy.iterator.SWIFTGalaxies.map` function can be used to apply a function to the collection of :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s represented by the :class:`~swiftgalaxy.iterator.SWIFTGalaxies` object (one at a time), and obtain the results. As a simple example:

.. code-block:: python

   from swiftgalaxy import SWIFTGalaxies, SOAP

   # define the function that we will apply to each SWIFTGalaxy object:
   def dm_median_position(sg):
       return np.median(sg.dark_matter.coordinates, axis=0)

   sgs = SWIFTGalaxies(
       "my_snapshot.hdf5",
       SOAP(
           "my_soap.hdf5",
           soap_index=[11, 22, 33],
       ),
   )
   my_result = sgs.map(dm_median_position)

The ``my_result`` variable will be a list containing the result of the ``dm_median_position`` function applied to the galaxies at ``soap_index=11``, ``22`` and ``33`` in that order. If the list ``[22, 33, 11]`` was instead passed to the :class:`~swiftgalaxy.halo_catalogues.SOAP` (or other halo catalogue class from :mod:`swiftgalaxy`), the output order would change correspondingly (even though internally the galaxies are most likely iterated over in the same order in both cases). Using :func:`~swiftgalaxy.iterator.SWIFTGalaxies.map` is in general not equivalent to:

.. code-block:: python

   my_result = [dm_median_position(sg) for sg in sgs]

Relying on the iteration order ``for sg in sgs`` should be avoided if the order of iteration or output matters, but can be appropriate if order is unimportant (such as a function or operation that produces an output file for each input :class:`~swiftgalaxy.reader.SWIFTGalaxy` named according to its unique identifier).

The :func:`~swiftgalaxy.iterator.SWIFTGalaxies.map` function can also accept additional argument values and/or keyword argument values. For example, the following code:

.. code-block:: python

   sg1 = SWIFTGalaxy(...)
   sg2 = SWIFTGalaxy(...)
   my_result = [my_func(sg1, 123, extra_data=456), my_func(sg2, 789, extra_data=None)]

Can be more succinctly written as (and will run more efficiently as):

.. code-block:: python

   sgs = SWIFTGalaxies(...)  # contains the same galaxies as sg1 and sg2, in that order
   my_result = sgs.map(
      my_func,
      args=[(123, ), (789, )],
      kwargs=[dict(extra_data=456), dict(extra_data=None)]
   )

Notice especially that the argument lists are bundled in :obj:`tuple`'s (of one element, in this case). The comma in ``(123, )`` is therefore not optional. If the function accepted two arguments they could be passed as something like ``args=[(123, "abc"), (456, "def")]``. Additional keyword arguments can similarly be added by adding additional :obj:`dict` entries.
   
Parallel iteration
------------------

There is an obvious opportunity to support iterating over :class:`~swiftgalaxy.reader.SWIFTGalaxy` objects in parallel through the :class:`~swiftgalaxy.iterator.SWIFTGalaxies` interface. The initial release of the :class:`~swiftgalaxy.iterator.SWIFTGalaxies` feature has omitted this to focus on a working serial implementation first. Tools for parallel analysis are planned for a future release.

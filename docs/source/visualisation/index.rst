Visualisation
=============

:class:`~swiftgalaxy.reader.SWIFTGalaxy` objects behave just like :class:`~swiftsimio.reader.SWIFTDataset` objects for the purposes of using the :mod:`~swiftsimio.visualisation` tools. Keep in mind that a :class:`~swiftgalaxy.reader.SWIFTGalaxy` has usually applied a mask to particles and has recentered and/or transformed the coordinate frame. The mask is one reason that it is usually best to set ``periodic=False`` when calling visualisation routines.

.. warning::
   The :mod:`swiftsimio` visualisation tools do not correctly handle periodic boundaries when coordinates have been rotated. The visualisation tools cannot currently detect whether a :class:`~swiftgalaxy.reader.SWIFTGalaxy` object has been rotated, and can give incorrect results if periodic boundaries are used. It is therefore recommended to always set ``periodic=False`` in all calls to :mod:`swiftsimio` visualisation routines when passing a :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

When a :class:`~swiftgalaxy.reader.SWIFTGalaxy` has been recentered (the default behaviour) the visualisation ``region`` should be given relative to the centre, not the box coordinates. For example:

.. code-block:: python

   from swiftsimio.visualisation.projection import project_gas
   sg = SWIFTGalaxy(..., auto_recenter=True)  # the default
   image = project_gas(
       sg,
       project="masses",
       parallel=True,
       periodic=False,
       resolution=256,
       region=cosmo_array(
           [-0.5, 0.5, -0.5, 0.5],
	       u.Mpc,
	       comoving=True,
	       scale_factor=sg.metadata.a,
	       scale_exponent=1,
       )
   )

For general information on the visualisation routines consult the :mod:`swiftsimio.visualisation` documentation pages.

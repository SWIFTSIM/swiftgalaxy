---
title: "SWIFTGalaxy"
tags:
  - Python
  - astronomy
  - simulations
authors: 
  - name: Kyle A. Oman
    orcid: 0000-0001-9857-7788
    affiliation: "1, 2"
    corresponding: true
affiliations:
  - name: Institute for Computational Cosmology, Physics Department, Durham University
    index: 1
  - name: Centre for Extragalactic Astronomy, Physics Department, Durham University
    index: 2
date: 4 August 2024
codeRepository: https://github.com/SWIFTSIM/swiftgalaxy
license: LGPLv3
bibliography: bibliography.bib
---

# Summary

SWIFTGalaxy is an open-source astrophysics module that extends SWIFTSimIO [@Borrow2020] tailored to analyses of particles belonging to individual galaxies simulated with SWIFT [@Schaller2024]. It inherits from and extends the functionality of the SWIFTSimIO's SWIFTDataset class. It understands the content of halo catalogues (supported: Velociraptor [@Elahi2019], [Caesar](https://github.com/dnarayanan/caesar), SOAP [@McGibbon2025]) and therefore which particles belong to a galaxy or other group of particles, and its integrated properties. The particles occupy a coordinate frame that is enforced to be consistent, such that particles loaded on-the-fly will match e.g. rotations and translations of particles already in memory. Intuitive masking of particle datasets is also enabled. Utilities to make working in cylindrical and spherical coordinate systems more convenient are also provided. Finally, tools to iterate efficiently over multiple galaxies are provided.

# Background

Cosmological hydrodynamical galaxy formation simulations begin with a representation of the early universe where its two key constituents - dark matter and gas - are discretized. The dark matter is usually represented with collisionless particles that respond only to the gravitational force, while gas may be represented by particles (Lagrangian tracers) or mesh cells (Eulerian tracers) and obeys the laws of hydrodynamics in addition to gravity. The SWIFT [@Schaller2024] code takes the former approach, implementing the smoothed particle hydrodynamics [SPH, @Lucy1977; @Gingold1977] formalism. Observations of the cosmic microwave background are used to constrain a multiscale Gaussian random field to formulate the initial conditions for a simulation of a representative region of the early Universe, before the first galaxies formed [e.g. @Bertschinger2001]. In addition to hydrodynamics and gravity, galaxy formation models include additional physical processes such as radiative gas cooling, star formation, energetic feedback from supernovae and supermassive black hole accretion, and more. Many of these are formally unresolved, leading these to be collectively refered to as "sub-grid" elements of the models.

As time integration of a cosmological hydrodynamical galaxy formation simulation proceeds, outputs containing a "snapshot" of the simulation state are typically written at regular intervals. These contain tables of the properties of particles including at minimum positions, velocities, masses and unique identifiers. More physically complex particle types, such as stars and gas, carry additional properties such as temperature, metallicity, internal energy, and many more. SWIFT snapshot files also include rich metadata recording the full configuration of the code and galaxy formation model, plus physical units and related information for all data.

For many types of analysis the snapshot files alone are insufficient: knowledge of which groups of particles "belong" to gravitationally bound structures - galaxies - is needed. In SWIFT this is most commonly determined in a two-stage process. In the first stage particles are linked to neigbouring particles separated by less than a suitably chosen linking length to form connected "friends-of-friends" [FOF; @Davis1985] groups. The second step consists of finding overdense, self-bound particle groups or "halos". There are numerous algorithms that accomplish this task; the current implementation recommended for the SWIFT community is the HBT-Herons [@ForouharMoreno2025] implementation of the hierarchical bound-tracing plus [HBT+; @Han2018]. Finally, the properties of halos can be tabulated in "halo catalogues" - again many algorithms exist. This is often done as part of the halo finding process, but the recommendation for the SWIFT community is to move this to a separate step and use the spherical overdensity aperture properties [SOAP; @McGibbon2025] tool.

# Statement of need

The collection of standard cosmological hydrodynamical simulation data products - snapshots of particles, halo membership information and halo catalgues - leads to a generic workflow to begin analysis of individual galaxies in such simulations. First, an object of interest is identified in the halo catalogue. Then, the halo membership information is queried to locate its member particles. Finally, the particles are loaded from the snapshot to proceed with analysis. The SWIFT community already maintains the SWIFTSimIO tool [@Borrow2020] that supports reading in SWIFT snapshot files and SOAP catalogue files. It uses metadata in the files to annotate data arrays with physical units and relevant cosmological information. It is also able to efficiently select spatial sub-regions of a simulation by taking advantage of the fact the particles in cells of a "top-level cell" grid covering the simulation domain are stored contiguously. Finally, it includes some data visualisation tools.

SWIFTGalaxy extends SWIFTSimIO by implementing the workflow outlined above supplemented with additional features including coordinate transformations, data array masking and efficient iteration over galaxies; all of the SWIFTSimIO features, including use of the visualisation tools, are inherited by SWIFTGalaxy. This is intended to avoid duplication of effort by users who would otherwise each need to implement the same steps independently, and helps to ensure that the implementation provided is optimal. It also helps to avoid common errors, such as applying a coordinate transformation to one particle type (e.g. rotation of the dark matter) but forgetting to apply it to others (e.g. the gas remains unrotated) by enforcing a consistent data state where applicable. A core design principle of SWIFTGalaxy is that only operations with an unambiguous implementation are included - in other words, SWIFTGalaxy tries to avoid making any decisions for its users. This is a key difference when compared to other packages serving this purpose, such as pynbody [@Pontzen2013] and yt [@Turk2011] that can also be used with SWIFT data sets. Other packages are also less tailored to SWIFT and therefore less able to take advantage of the detailed structure of the data layout on disk and information available as metadata.

SWIFTGalaxy is hosted on [GitHub](https://github.com/SWIFTSIM/swiftgalaxy) and has documentation available through [ReadTheDocs](https://swiftgalaxy.readthedocs.io). The first research article using the code has recently appeared [@Trayford2025]; many more projects using the code are currently ongoing, especially in the context of the COLIBRE simulations project (Schaye et al., in preparation; Chaikin et al., in preparation).

# Acknowledgements

KAO acknowledges support by the Royal Society trhough Dorothy Hodgkin Fellowship DHF/R1/231105, by STFC through grant ST/T000244/1, and by the European Research Council (ERC) through an Advanced Investigator Grant to C. S. Frenk, DMIDAS (GA 786910). This work has made use of NASA's Astrophysics Data System.

# References

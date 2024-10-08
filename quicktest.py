from swiftgalaxy import SWIFTGalaxy
from swiftgalaxy.halo_finders import Caesar
import matplotlib
import matplotlib.pyplot as plt
import unyt as u
from swiftsimio.visualisation.projection import project_gas, project_pixel_grid
from matplotlib.backends.backend_pdf import PdfPages

sg = SWIFTGalaxy(
    "./simba_s12.5n128_0012.hdf5",
    Caesar("./s12.5n128_0012.hdf5", group_type="galaxy", group_index=0),
)


def myvis(sg):
    aperture_radius = 50 * u.kpc
    region = [-aperture_radius, aperture_radius, -aperture_radius, aperture_radius]
    gas_map = project_gas(
        sg,
        resolution=256,
        project="masses",
        parallel=True,
        region=region,
    )
    star_map = project_pixel_grid(
        data=sg.stars,
        boxsize=sg.metadata.boxsize,
        resolution=256,
        project="masses",
        parallel=True,
        region=region,
    )
    with PdfPages("test.pdf") as pdffile:
        fig = plt.figure(1, figsize=(6, 3))
        sp1, sp2 = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
        sp1.imshow(
            matplotlib.colors.LogNorm()(gas_map.value),
            cmap="viridis",
            extent=region,
        )
        sp1.set_xlabel(f"x [{aperture_radius.units}]")
        sp1.set_ylabel(f"y [{aperture_radius.units}]")
        sp1.text(0.9, 0.9, "gas", ha="right", va="top", transform=sp1.transAxes)
        sp2.imshow(
            matplotlib.colors.LogNorm()(star_map),
            cmap="magma",
            extent=region,
        )
        sp2.set_xlabel(f"x [{aperture_radius.units}]")
        sp2.set_ylabel(f"y [{aperture_radius.units}]")
        sp2.text(0.9, 0.9, "stars", ha="right", va="top", transform=sp2.transAxes)
        fig.subplots_adjust(wspace=0.4)
        fig.savefig("test.pdf", format="pdf")
        fig = plt.figure(1, figsize=(6, 3))
        sp1, sp2 = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
        sp1.imshow(
            matplotlib.colors.LogNorm()(gas_map.value),
            cmap="viridis",
            extent=region,
        )
        sp1.set_xlabel(f"x [{aperture_radius.units}]")
        sp1.set_ylabel(f"y [{aperture_radius.units}]")
        sp1.text(0.9, 0.9, "gas", ha="right", va="top", transform=sp1.transAxes)
        sp2.imshow(
            matplotlib.colors.LogNorm()(star_map),
            cmap="magma",
            extent=region,
        )
        sp2.set_xlabel(f"x [{aperture_radius.units}]")
        sp2.set_ylabel(f"y [{aperture_radius.units}]")
        sp2.text(0.9, 0.9, "stars", ha="right", va="top", transform=sp2.transAxes)
        fig.subplots_adjust(wspace=0.4)
        fig.savefig(pdffile, format="pdf")
        fig = plt.figure(2, figsize=(6, 3))
        cunit = u.kpc
        vunit = u.km / u.s
        sp1, sp2 = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
        sp1.plot(
            sg.gas.cartesian_coordinates.x.to(cunit),
            sg.gas.cartesian_velocities.x.to(vunit),
            ",k",
            rasterized=True,
        )
        sp1.set_xlabel(f"x [{cunit}]")
        sp1.set_ylabel(f"vx [{vunit}]")
        sp1.text(0.9, 0.9, "gas", ha="right", va="top", transform=sp1.transAxes)
        sp2.plot(
            sg.stars.cartesian_coordinates.x.to(cunit),
            sg.stars.cartesian_velocities.x.to(vunit),
            ",k",
            rasterized=True,
        )
        sp2.set_xlabel(f"x [{cunit}]")
        sp2.set_ylabel(f"vx [{vunit}]")
        sp2.text(0.9, 0.9, "stars", ha="right", va="top", transform=sp2.transAxes)
        fig.subplots_adjust(wspace=0.4)
        fig.savefig(pdffile, format="pdf")


myvis(sg)

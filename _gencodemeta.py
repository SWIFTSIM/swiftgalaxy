import os


class Lines(list):
    ind = "    "

    def __init__(self, *args, **kwargs):
        self.nind = 0
        super().__init__(*args, **kwargs)
        return

    def indent(self):
        self.nind += 1
        return

    def unindent(self):
        self.nind -= 1
        return

    def append(self, line):
        super().append(self.nind * self.ind + line + "\n")
        return


def gencodemeta():
    with open(
        os.path.join(os.path.dirname(__file__), "swiftgalaxy", "__version__.py")
    ) as version_file:
        version = version_file.read().split("=")[-1].strip()[1:-1]

    fields = {
        "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
        "@type": "SoftwareSourceCode",
        "name": "SWIFTGalaxy",
        "description": "SWIFTGalaxy provides a software abstraction of simulated "
        "galaxies produced by the SWIFT smoothed particle hydrodynamics code. It "
        "extends the SWIFTSimIO module and is tailored to analyses of particles "
        "belonging to individual simulated galaxies. It inherits from and extends the "
        "functionality of the SWIFTDataset. It understands the output of halo finders "
        "and therefore which particles belong to a galaxy, and its integrated "
        "properties. The particles occupy a coordinate frame that is enforced to be "
        "consistent, such that particles loaded on-the-fly will match e.g. rotations "
        "and translations of particles already in memory. Intuitive masking of particle "
        "datasets is also enabled. Finally, some utilities to make working in "
        "cylindrical and spherical coordinate systems more convenient are also provided.",
        "identifier": "",
        "author": [
            {
                "@type": "Person",
                "givenName": "Kyle A.",
                "familyName": "Oman",
                "@id": "0000-0001-9857-7788",
            }
        ],
        "citation": "",
        "relatedLink": ["https://pypi.org/project/swiftgalaxy"],
        "codeRepository": ["https://github.com/SWIFTSIM/swiftgalaxy"],
        "version": version,
        "license": "https://spdx.org/licenses/GPL-3.0-only.html",
    }

    L = Lines()

    L.append("{")
    L.indent()
    for k, v in fields.items():
        if isinstance(v, str):
            L.append('"{:s}": "{:s}",'.format(k, v))
        elif isinstance(v, list):
            L.append('"{:s}": ['.format(k))
            L.indent()
            for line in v:
                if isinstance(line, str):
                    L.append('"{:s}",'.format(line))
                elif isinstance(line, dict):
                    L.append("{")
                    L.indent()
                    for kk, vv in line.items():
                        L.append('"{:s}": "{:s}",'.format(kk, vv))
                    L.unindent()
                    L.append("},")
                else:
                    raise RuntimeError("Unhandled!")
            L.unindent()
            L.append("],")
        else:
            raise RuntimeError("Unhandled!")
    L.unindent()
    L.append("}")

    with open("codemeta.json", "w") as f:
        f.writelines(L)

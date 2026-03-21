# https://kwant-project.org/doc/dev/tutorial/kpm
# https://kwant-project.org/doc/1/tutorial/graphene

# necessary imports
import kwant
import numpy as np


# define the system
def make_syst(r=30, t=-1, a=1):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1)

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.shape(circle, (0, 0))] = 0.
    syst[lat.neighbors()] = t
    syst.eradicate_dangling()

    return syst

fsyst = make_syst().finalized()
spectrum = kwant.kpm.SpectralDensity(fsyst)

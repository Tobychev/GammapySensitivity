#!/usr/bin/env python
# coding: utf-8

# # Gammapy imports


from gammapy.estimators import SensitivityEstimator
from gammapy.data import FixedPointingInfo, Observation, observatory_locations
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from gammapy.irf import (
    load_irf_dict_from_file,
    RadMax2D,
    EnergyDispersion2D,
    Background3D,
    Background2D,
)
from astropy.wcs.utils import proj_plane_pixel_area

from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion, PointSkyRegion
from pathlib import Path


# # Further imports

# In[2]:


import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from astropy.visualization import quantity_support

quantity_support()


# # Sensitivity function
class RadMaxGeom(RegionGeom):
    def __init__(
        self,
        region,
        rad_max,
        radii=None,
        pointing=None,
        axes=None,
        wcs=None,
        binsz_wcs="0.1 deg",
    ):
        covering_region = CircleSkyRegion(
            center=region.center, radius=rad_max.data.max() * rad_max.unit
        )
        super().__init__(covering_region, axes=axes, wcs=wcs, binsz_wcs=binsz_wcs)

        if isinstance(rad_max, RadMax2D):
            self._rad_max = rad_max
        else:
            raise ValueError(
                f"rad_max needs to be instance of `RadMax2d`, got {type(rad_max)}"
            )

        self._radii = radii
        self._pointing = pointing

    @classmethod
    def create(cls, region, rad_max, **kwargs):
        return cls.from_regions(regions=region, rad_max=rad_max, **kwargs)

    def setup_for_solid_angle(self, pointing, obs):
        point_icrs = pointing.get_icrs(obs.tmid)
        reg_coord = self.to_image().get_coord()
        sky_coord = reg_coord.skycoord
        offset = sky_coord.separation(point_icrs)
        self._pointing = point_icrs
        if self._rad_max.axes["offset"].nbin > 1:
            eval_en = self.axes["energy"].center
        else:
            eval_en = self.axes["energy"].center[:, np.newaxis]

        self._radii = np.squeeze(self._rad_max.evaluate(offset=offset, energy=eval_en))

    def solid_angle(self):
        if self.region is None:
            raise ValueError("Region definition required.")

        areas = np.zeros_like(self._radii.value)
        # Extra axes to match dimensions from a RegionGeom
        areas = areas[:, np.newaxis, np.newaxis]
        for idx, r in enumerate(self._radii):
            area = CircleSkyRegion(self._pointing, radius=r).to_pixel(self.wcs).area
            areas[idx] = area

        d_omegas = areas * proj_plane_pixel_area(self.wcs) * u.deg**2
        return d_omegas.to("sr")


def gp_sensitivity_curve(irfs, root_e_bins, enclosure=True, conf={}):

    energy_axis_reco = MapAxis.from_nodes(
        root_e_bins, interp="log", name="energy"
    ).to_node_type("edges")
    energy_axis_true = MapAxis.from_energy_bounds(
        conf.get("E_true_min", 0.01 * u.TeV),
        conf.get("E_true_max", 350 * u.TeV),
        nbin=conf.get("E_true_nbin", 150),
        name="energy_true",
    )
    pointing = SkyCoord(
        ra=conf.get("point_ra", 0 * u.deg), dec=conf.get("point_dec", 0 * u.deg)
    )
    on_region_radius = conf.get("on_region", 0.1 * u.deg)
    offset = conf.get("point_offset", 0.5 * u.deg)
    offset_dir = conf.get("point_offset_dir", 0.0 * u.deg)

    pointing_info = FixedPointingInfo(fixed_icrs=pointing)
    source_position = pointing.directional_offset_by(offset_dir, offset)
    location = observatory_locations["cta_south"]
    livetime = conf.get("livetime", 50.0 * u.h)

    obs = Observation.create(
        pointing=pointing_info, irfs=irfs, livetime=livetime, location=location
    )

    if enclosure:
        on_region = CircleSkyRegion(source_position, radius=on_region_radius)
        correct_containment = False
        geom = RegionGeom.create(region=on_region, axes=[energy_axis_reco])
    else:
        on_region = PointSkyRegion(source_position)
        geom = RadMaxGeom.create(
            region=on_region, rad_max=irfs["rad_max"], axes=[energy_axis_reco]
        )
        geom.setup_for_solid_angle(pointing_info, obs)
        correct_containment = False

    empty_dataset = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

    spectrum_maker = SpectrumDatasetMaker(
        selection=["exposure", "edisp", "background"],
        containment_correction=correct_containment,
    )
    dataset = spectrum_maker.run(empty_dataset, obs)
    on_radii = None
    if enclosure:
        containment = 0.68
        dataset.exposure *= containment
        on_radii = obs.psf.containment_radius(
            energy_true=energy_axis_reco.center, offset=offset, fraction=containment
        )
        factor = (1 - np.cos(on_radii)) / (1 - np.cos(on_region_radius))
        dataset.background *= factor.value.reshape((-1, 1, 1))

    dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
        dataset=dataset, acceptance=1, acceptance_off=5
    )
    sensitivity_estimator = SensitivityEstimator(
        gamma_min=10,
        n_sigma=5,
        bkg_syst_fraction=0.05,
    )
    gp_sensitivity_table = sensitivity_estimator.run(dataset_on_off)
    return (
        gp_sensitivity_table["e_ref"],
        gp_sensitivity_table["e2dnde"],
        (gp_sensitivity_table, on_radii),
    )


# # Load functions

# In[4]:


def load_sensitivity(root_file):
    irf_root = uproot.open(root_file)
    sens, sens_ebins = irf_root["DiffSens"].to_numpy()
    sens_ebins = 10**sens_ebins * u.TeV
    sens = sens * u.erg / u.s / u.cm**2
    return sens, sens_ebins


def load_irf_with_rad_max(irf_file, root_file):
    irf_root = uproot.open(root_file)
    theta, log_E = irf_root["ThetaCut;1"].to_numpy()
    Eaxes = MapAxis(10**log_E, name="energy", unit="TeV")
    offset = MapAxis([0, 5], name="offset", unit="degree")

    irf_table = load_irf_dict_from_file(irf_file)

    irf_table["rad_max"] = RadMax2D(
        data=theta[..., np.newaxis],
        axes=[Eaxes, offset],
        unit="deg",
        interp_kwargs={"method": "nearest", "fill_value": None},
    )
    return irf_table


# # Load irfs


IRF_ROOT = Path("./IRFs/")

prod3_irf_fits = IRF_ROOT / "Prod3_South_z20_S_50h_irf_file.fits"
prod3_irf_root = IRF_ROOT / "CTA-Performance-prod3b-v2-South-20deg-S-50h.root"

prod5_irf_fits = (
    IRF_ROOT / "Prod5-South-20deg-SouthAz-14MSTs37SSTs.180000s-v0.1.fits.gz"
)
prod5_irf_root = IRF_ROOT / "Prod5-South-20deg-SouthAz-14MSTs37SSTs.180000s-v0.1.root"

prod5_irf_point_fits = (
    IRF_ROOT / "Prod5-South-20deg-SouthAz-14MSTs37SSTs.180000s_pointlike.fits"
)

pyirf_irf_fits = IRF_ROOT / "Match_ED_IRF.fits"


# prod5_sens, p5_en = load_sensitivity(prod5_irf_root)
irf_tab_p5 = load_irf_dict_from_file(prod5_irf_fits)
irf_tab_p5_point = load_irf_dict_from_file(prod5_irf_point_fits)
irf_tab_py = load_irf_dict_from_file(pyirf_irf_fits)


# prod3_sens, p3_en = load_sensitivity(prod3_irf_root)
# irf_tab_p3 = load_irf_with_rad_max(prod3_irf_fits, prod3_irf_root)

irf_tab_py = load_irf_dict_from_file(pyirf_irf_fits)

reco_energy_bins = [
    0.01260,  # Truncated to keep within available rad_max table
    0.0199526231496888,
    0.03162277660168379,
    0.05011872336272722,
    0.07943282347242814,
    0.12589254117941667,
    0.1995262314968879,
    0.3162277660168378,
    0.501187233627272,
    0.7943282347242809,
    1.2589254117941662,
    1.9952623149688768,
    3.162277660168376,
    5.011872336272719,
    7.943282347242805,
    12.58925411794165,
    19.95262314968877,
    31.62277660168376,
    50.118723362727145,
    79.43282347242797,
    125.89254117941648,
    199.5259,  # Took it down a bit to match range in tool irf
] * u.TeV

py_gE, py_gSens, (res_py, radii) = gp_sensitivity_curve(
    irf_tab_py,
    #  irf_tab_p5_point,
    reco_energy_bins,
    enclosure=False,
    conf={"point_offset": 0.5 * u.deg},
)

p5p_gE, p5p_gSens, (res, radii) = gp_sensitivity_curve(
    irf_tab_p5_point,
    reco_energy_bins,
    enclosure=False,
    conf={"point_offset": 0.5 * u.deg},
)

print(res)
sens_store = np.load("Prod5_sensitivity.npz")
prod5_sens, prod5_energy = sens_store["prod5_sens"], sens_store["prod5_energy"]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plt.step(res_py["e_ref"], res_py["e2dnde"], label="Tool + gammapy", lw=2)
#  plt.step(res["e_ref"], res["e2dnde"], label="Prod5 + gammapy")
plt.stairs(
    prod5_sens[1:],
    prod5_energy[1:],
    label="Prod5 root 'DiffSens'",
    baseline=None,
    lw=2,
    linestyle="dashed",
    alpha=0.4,
    color="k",
)
plt.title("Prod 5")

plt.ylim(4e-15, 8e-8)
plt.loglog()
plt.legend()
plt.show()

#!/usr/bin/env python3
# Boston-centered range rings (Azimuthal Equidistant projection) with geodesic plotting.

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
import matplotlib.patheffects as pe

plt.style.use('seaborn-v0_8-deep')

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'


# ---------------- Hub & ranges ----------------
CITY = "BOSTON"
LAT0, LON0 = 42.3656, -71.0096  # KBOS
RINGS_NMI = {
    "2000 nmi": 2000,
    "3000 nmi": 3000,
    "6000 nmi": 6000,
}

#COLORS = {
#    "2000 nmi": "#2043ff",
#    "3000 nmi": "#f4c430",
#    "6000 nmi": "#ff7f0e",
#}

COLORS = {
    "2000 nmi": "black",
    "3000 nmi": "black",
    "6000 nmi": "black",
}
WIDTHS = {k: 2.5 for k in RINGS_NMI}

NMI_TO_M = 1852.0
geod = Geod(ellps="WGS84")

def geodesic_circle(lon0, lat0, radius_m, n=721):
    """Return lon/lat points of a geodesic circle around (lon0,lat0)."""
    az = np.linspace(0, 360, n)
    lons, lats, _ = geod.fwd(
        np.full_like(az, lon0),
        np.full_like(az, lat0),
        az,
        np.full_like(az, radius_m),
    )
    return lons, lats

# ---------------- Figure ----------------
proj = ccrs.AzimuthalEquidistant(central_longitude=LON0, central_latitude=LAT0)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=proj)
ax.set_global()
ax.coastlines(resolution="50m", linewidth=1.0, color="gray")

ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.01, edgecolor = "lightgray")
ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.01, edgecolor="lightgray")
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor="C0", alpha = 0.3, edgecolor="none")
ax.add_feature(cfeature.LAND.with_scale("50m"),  facecolor="#e9eef2", edgecolor="none")
#ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.01, edgecolor="none", color = "lightgray")

# Basemap
#ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#91a2b1", edgecolor="none")
#ax.add_feature(cfeature.LAND.with_scale("110m"),  facecolor="#e9eef2", edgecolor="none")
#ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.4, edgecolor="#8793a1")
#ax.gridlines(draw_labels=False, linewidth=0.4, color="#aab3bf", alpha=0.6)


WIDTHS = {
    "2000 nmi": 1.5,
    "3000 nmi": 2.0,
    "7000 nmi": 2.5,
}

# Hub marker + label
#ax.plot(LON0, LAT0, marker="o", markersize=5, color="#0b1b3b", transform=ccrs.PlateCarree(), zorder=10)
#ax.text(LON0, LAT0 - 7, CITY, transform=ccrs.PlateCarree(),
#        ha="center", va="center", fontsize=18, weight="bold", color="#0b1b3b", zorder=10)

# Range rings (use Geodetic transform so dateline is handled properly)
for label, dist_nmi in RINGS_NMI.items():
    r_m = dist_nmi * NMI_TO_M
    lons, lats = geodesic_circle(LON0, LAT0, r_m)
    ax.plot(lons, lats,
            transform=ccrs.Geodetic(),
            color=COLORS.get(label, "k"),
            linewidth=WIDTHS.get(label, 2.5),  # pick per ring
            solid_capstyle="round",
            zorder=5)

azimuths = [120, 130, 140]  # one per ring, tweak as needed

#for (label, dist_nmi), az in zip(RINGS_NMI.items(), azimuths):
#    r_m = dist_nmi * NMI_TO_M
#    lon_txt, lat_txt, _ = geod.fwd(LON0, LAT0, az, r_m)
#    ax.text(lon_txt, lat_txt, label,
#            transform=ccrs.PlateCarree(),
#            fontsize=12, weight="bold", color=COLORS.get(label, "k"),
#            ha="left", va="center", zorder=6)

LABEL_OFFSET_NMI = 225  # try 15–40 nmi
for (label, dist_nmi), az in zip(RINGS_NMI.items(), azimuths):
    r_m = dist_nmi * NMI_TO_M
    r_label = r_m + LABEL_OFFSET_NMI * NMI_TO_M  # push outward
    lon_txt, lat_txt, _ = geod.fwd(LON0, LAT0, az, r_label)
    ax.text(lon_txt, lat_txt, label, transform=ccrs.PlateCarree(),
            fontsize=12, weight="bold", color=COLORS.get(label, "k"),
            ha="left", va="center", zorder=6,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            clip_on=False)


if 'geo' in ax.spines:
    ax.spines['geo'].set_visible(False)
    
    
plt.tight_layout()
# To save instead of show:
# plt.savefig("boston_range_aeqd.png", dpi=300, bbox_inches="tight")
plt.savefig('Range_Diagram.png', dpi=300, transparent=False)
#plt.show()

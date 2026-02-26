"""Configuration constants for the multi-variable satellite pickler."""

# Geographical Bounding Box - Yucatan Peninsula
LON_MIN, LON_MAX = -92.20216, -85.61256
LAT_MIN, LAT_MAX = 19.57871, 22.86906

# Binning parameters (consistent spatial grid for all variables)
LAT_BINS = 250
LON_BINS = 250

# Time range
START_YEAR = 2010
END_YEAR = 2019

# Data directories (patterns with {year} placeholder)
MUR_SST_PATTERN = r"E:\satdata\sst\MUR-JPL-L4-GLOB-v4.1_Yucatan Peninsula_{year}-01-01_{year}-12-31"
MODIS_OC_PATTERN = r"G:\satdata\cdom\Yucatan Peninsula_{year}-01-01_{year}-12-31"
SSL_PATTERN = r"E:\satdata\GCOOS_Yucatan Peninsula_2010-01-01_2019-12-31"

# Bathymetry data
BATHYMETRY_FILE = r"E:\geodata\gebco_2024_sub_ice_topo\GEBCO_2024_sub_ice_topo.nc"
BATHYMETRY_VAR = "elevation"

# Output
OUTPUT_DIR = r"E:\satdata\Custom"
OUTPUT_FILENAME = "yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v3.pkl"

# Variable names
SST_VAR = "analysed_sst"
CHLOR_VAR = "chlor_a"
RRS_412_VAR = "Rrs_412"
RRS_555_VAR = "Rrs_555"
RRS_443_VAR = "Rrs_443"
RRS_547_VAR = "Rrs_547"
SSL_VAR = "adt"

# CDOM calculation constants
CDOM_B0 = 0.2487
CDOM_B1 = 14.028
CDOM_B2 = 4.085

# CDOM spectral slope constants (Mannino et al.)
# Ln[S_275:295] = -3.258 + 0.336*Ln[Rrs_443] - 0.279*Ln[Rrs_547]
# Ln[S_300:600] = -3.640 + 0.186*Ln[Rrs_443] - 0.146*Ln[Rrs_547]
S275_295_COEFFS = {"B0": -3.258, "B1": 0.336, "B2": -0.279}
S300_600_COEFFS = {"B0": -3.640, "B1": 0.186, "B2": -0.146}

# Quality control thresholds
MIN_SST_CELSIUS = -2.0
MAX_SST_CELSIUS = 40.0
MIN_CHLOR = 0.01
MAX_CHLOR = 100.0
MIN_CDOM = 0.0
MAX_CDOM = 1.0
MIN_SSL = -2.0
MAX_SSL = 2.0

# SSL native resolution (informational)
SSL_NATIVE_RES_DEG = 0.25

# NASA MUREP Local Project Overview

This repository contains three distinct components, each focused on a different aspect of satellite data analysis and environmental monitoring. The parts are independent and serve different scientific or technical purposes.

---

## 1. `river_plumes`

**Purpose:**  
This folder contains notebooks and documentation for analyzing river plume dynamics in the Northern Gulf of Mexico, with a focus on calculating Total Suspended Solids (TSS) using MODIS Aqua satellite imagery.

**Key Files:**
- `mississippi_tss.ipynb`: Jupyter notebook for TSS analysis in the Mississippi River plume.
- `tss.md`: Documentation summarizing empirical models for TSS estimation from satellite data.
- `download.ipynb`: Automated download and organization of MODIS Aqua data for river plume studies.

**Highlights:**
- Implements empirical and semi-analytical models for TSS retrieval.
- Provides region-specific guidance for atmospheric correction and data processing.
- Supports multi-period analysis for environmental change detection.

---

## 2. `cdom_sst`

**Purpose:**  
This folder focuses on the study of Colored Dissolved Organic Matter (CDOM) and Sea Surface Temperature (SST) in coastal and shelf regions using satellite data.

**Key Files:**
- `download.ipynb`: Automated download of MODISA L2 Ocean Color and SST datasets.
- `test_cdom.ipynb`, `test_cdomsst.ipynb`, `test_cdomsstanom.ipynb`, `test_sst.ipynb`, `test_sstanom.ipynb`, `test.ipynb`: Notebooks for processing, analyzing, and visualizing CDOM and SST data.
- `README.md`: Detailed overview of algorithms and workflows for CDOM/SST analysis.

**Highlights:**
- GPU-accelerated extraction and interpolation of satellite data.
- Region and time filtering for targeted environmental studies.
- Advanced visualization and combination of CDOM and SST metrics.

---

## 3. `cdom_measurements`

**Purpose:**  
This folder contains tools for analyzing in-situ CDOM measurements, including statistical summaries and spatial mapping.

**Key Files:**
- `analyze.ipynb`: Notebook for loading, processing, and visualizing CDOM absorbance data by season and location.

**Highlights:**
- Seasonal statistical analysis of CDOM measurements.
- Geospatial visualization of measurement sites and values.
- Integration with external data loading modules.

---

Each part of the project is self-contained and can be used independently for its respective scientific or technical workflow.

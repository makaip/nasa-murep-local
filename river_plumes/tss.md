<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Empirical Models for Calculating Total Suspended Solids in the Northern Gulf of Mexico Using MODIS Aqua Satellite Imagery

The retrieval of Total Suspended Solids (TSS) concentrations from satellite imagery represents an important tool for monitoring water quality in coastal environments such as the Northern Gulf of Mexico. Based on the available literature, several empirical models have been developed that utilize MODIS Aqua spectral bands to estimate TSS concentrations. The most relevant model for the Northern Gulf of Mexico uses the ratio of remote sensing reflectance at specific wavelengths, particularly Rrs(670)/Rrs(555), which has been demonstrated to be an effective indicator of suspended particulate matter in these waters. However, comprehensive region-specific models with detailed constants are limited in the literature, with most approaches requiring adaptation from other regions or generalized frameworks.

## Remote Sensing Approaches for TSS Estimation

Remote sensing of Total Suspended Solids (also referred to as Total Suspended Matter or Suspended Particulate Matter) relies on the optical properties of water being altered by suspended particles. These particles scatter and absorb light in ways that can be detected by satellite sensors such as MODIS Aqua. Several approaches have been developed to convert satellite measurements into TSS concentrations.

### General Methodological Framework

The fundamental approach to TSS retrieval from satellite imagery involves establishing relationships between the satellite-measured reflectance and in-situ TSS measurements. These relationships typically involve either:

1. Single-band models that use reflectance from a specific wavelength
2. Band-ratio models that use the ratio of reflectances from two or more wavelengths
3. Multi-conditional models that apply different algorithms based on water classification
4. Semi-analytical models that incorporate inherent optical properties (IOPs) of water constituents

For coastal waters like the Northern Gulf of Mexico, which can range from moderately turbid to highly turbid, band-ratio approaches have proven particularly effective due to their ability to normalize for environmental variables that affect multiple wavelengths similarly[^5].

## Northern Gulf of Mexico Empirical Models

### D'Sa et al. (2007) Model

For the specific region of interest—the Northern Gulf of Mexico—research by D'Sa et al. (2007) demonstrated that the ratio of remote sensing reflectance at 670 nm to 555 nm provides a good indicator of suspended particulate matter concentrations[^5]. The model follows this general form:

TSS (mg/L) = a × [Rrs(670)/Rrs(555)]^b

Where:

- Rrs(670) is the remote sensing reflectance at 670 nm
- Rrs(555) is the remote sensing reflectance at 555 nm
- a and b are empirically derived constants

While the exact values of constants a and b for the Northern Gulf of Mexico are not specified in the available literature, this band ratio approach has been validated for this region[^5].

### Adaptable Empirical Models

Several other empirical models could potentially be adapted for the Northern Gulf of Mexico, based on successful applications in similar coastal environments:

#### Single-Band Models

Simple single-band models often use the red or near-infrared (NIR) bands of MODIS Aqua. The general form is:

TSS (g/m³) = A × Rrs(λ) + B

Where:

- Rrs(λ) is the remote sensing reflectance at wavelength λ (often 645 nm, 667 nm, or 678 nm for MODIS)
- A and B are empirical constants derived from in-situ measurements

Several researchers have employed this approach, including Miller \& McKee (2004) and Ondrusek et al. (2012)[^5].

#### Multi-Conditional Models

The SOLID (Statistical, inherent Optical property-based, and muLti-conditional Inversion proceDure) model represents a more sophisticated approach that could be applied to the Northern Gulf of Mexico. This model follows a three-step procedure:

1. Water-type classification based on input remote sensing reflectance
2. Retrieval of particulate backscattering in red or NIR regions
3. Estimation of TSS via water-type-specific empirical models[^2]

The SOLID model has shown superior performance compared to other algorithms, with improvements ranging from 10% to over 100% depending on water conditions[^2].

## Implementation Considerations for MODIS Aqua

MODIS Aqua provides suitable spectral bands for TSS retrieval in the Northern Gulf of Mexico. The key bands relevant for TSS models include:

1. Band 4 (545-565 nm, centered at 555 nm)
2. Band 1 (620-670 nm, centered at 645 nm)
3. Band 13 (662-672 nm, centered at 667 nm)
4. Band 14 (673-683 nm, centered at 678 nm)
5. Band 15 (743-753 nm, centered at 748 nm)

For optimal implementation of TSS retrieval models in the Northern Gulf of Mexico using MODIS Aqua data, several considerations should be taken into account:

### Atmospheric Correction

Proper atmospheric correction is crucial for accurate TSS retrieval. Research indicates that a 10% uncertainty in remote sensing reflectance (Rrs) leads to less than 20% uncertainty in TSS retrievals[^2]. When working with turbid coastal waters like those in the Northern Gulf of Mexico, standard atmospheric correction algorithms may need modification to account for non-zero reflectance in the near-infrared region.

### Spatial and Temporal Resolution

MODIS Aqua provides data at different spatial resolutions: 250 m, 500 m, and 1 km. For coastal applications in the Northern Gulf of Mexico, the 250 m or 500 m resolution bands are generally most appropriate. Many implementations use 14-day composites or monthly means to reduce the impact of cloud cover and other temporal variations[^1][^3].

### Water Type Classification

Turbidity conditions in the Northern Gulf of Mexico can vary widely, particularly near river mouths and following storm events. Conditional algorithms that apply different formulations based on water classification have been shown to perform better across a range of conditions. For example, Shen et al. (2010) employed different models for waters with low (<20 mg/L) and high (>20 mg/L) suspended particulate matter concentrations[^5].

## Limitations and Research Gaps

The available research on TSS models specifically calibrated for the Northern Gulf of Mexico has several limitations:

1. Most empirical models are site-specific and may not perform well when transferred to new regions without recalibration.
2. The constants in empirical formulas are often derived from limited in-situ datasets that may not represent the full range of conditions in the Northern Gulf of Mexico.
3. Seasonal and event-based (e.g., hurricanes, floods) variations can significantly affect the performance of empirical models.
4. The shallow bathymetry of parts of the Northern Gulf of Mexico can confound TSS retrieval due to bottom reflectance effects.
5. Current literature lacks comprehensive validation of models across the entire Northern Gulf of Mexico region.

The SOLID model, while promising for global application, would need specific validation and possibly recalibration for optimal performance in the Northern Gulf of Mexico[^2].

## Conclusion

Empirical models for calculating Total Suspended Solids in the Northern Gulf of Mexico using MODIS Aqua imagery primarily rely on the relationship between remote sensing reflectance ratios and in-situ TSS measurements. The most regionally specific approach identified is the model by D'Sa et al. (2007), which utilizes the Rrs(670)/Rrs(555) ratio as a proxy for suspended particulate matter. More sophisticated approaches like the SOLID model incorporate water-type classification and could potentially provide improved accuracy across varying conditions.

Future research should focus on developing comprehensive validation datasets for the Northern Gulf of Mexico and deriving region-specific constants for the empirical formulas. This would enhance the accuracy and reliability of satellite-derived TSS estimates for environmental monitoring, water quality assessment, and ecological studies in this important coastal region.

<div style="text-align: center">⁂</div>

[^1]: https://catalog.data.gov/dataset/sst-aqua-modis-npp-gulf-of-mexico-daytime-and-nighttime-11-microns-2002-2012-14-day-composite

[^2]: https://repository.library.noaa.gov/view/noaa/41928/noaa_41928_DS1.pdf

[^3]: http://data.europa.eu/89h/b245c405-f12f-40c7-85c9-ef03237679f8

[^4]: https://www.sciencedirect.com/science/article/abs/pii/S0034425715000887

[^5]: https://repository.library.noaa.gov/view/noaa/41849/noaa_41849_DS1.pdf

[^6]: https://modis.gsfc.nasa.gov/data/atbd/atbd_mod19.pdf

[^7]: https://www.sciencedirect.com/science/article/abs/pii/S0141113611001310

[^8]: https://ntrs.nasa.gov/citations/20205002804

[^9]: https://eastcoast.coastwatch.noaa.gov/cw_k490_hires.php

[^10]: https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1464942/full

[^11]: https://ikcest-drr.data.ac.cn/static/upload/c2/c233934a-bbdb-11e8-b94f-00163e0618d6.pdf

[^12]: https://modis.gsfc.nasa.gov/data/

[^13]: https://www.baydeltalive.com/assets/06942155460a79991fdf1b57f641b1b4/application/pdf/MODIS_Estimating_wide_range_Total_Suspended_Solids_concentrations_from_MODIS_250-m_imageries.pdf

[^14]: https://www.sciencedirect.com/science/article/abs/pii/S0034425718303390

[^15]: https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2017.00233/pdf

[^16]: https://modis.gsfc.nasa.gov/sci_team/meetings/200407/presentations/posters/ocean5.ppt

[^17]: https://www.tandfonline.com/doi/full/10.1080/01431161.2023.2240522

[^18]: https://ntrs.nasa.gov/api/citations/20110023407/downloads/20110023407.pdf

[^19]: https://ntrs.nasa.gov/api/citations/20210016749/downloads/article.pdf

[^20]: https://catalog.data.gov/dataset/sst-aqua-modis-npp-gulf-of-mexico-nighttime-11-microns-2002-2012-8-day-composite

[^21]: https://www.mdpi.com/2072-4292/10/7/987

[^22]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7730694/

[^23]: https://oceancolor.gsfc.nasa.gov/data/10.5067/AQUA/MODIS/L3B/RRS/2022

[^24]: https://www.sciencedirect.com/science/article/pii/S1470160X23006556

[^25]: https://cmr.earthdata.nasa.gov/search/concepts/C1615905765-OB_DAAC.html

[^26]: https://www.sciencedirect.com/science/article/abs/pii/S0301479721016121

[^27]: https://gce-lter.marsci.uga.edu/public/app/biblio_results.asp?Library=GCE\&SubjectMode=contains\&URLs=yes\&Order=year\&SortOrder=DESC\&ShowCount=no\&Abstracts=yes\&Format=\&PageTitle=

[^28]: https://www.mdpi.com/2072-4292/8/10/810

[^29]: https://www.mdpi.com/2073-4441/13/8/1078

[^30]: https://www.mdpi.com/2072-4292/8/7/556

[^31]: https://ouci.dntb.gov.ua/en/works/45aoGyk4/

[^32]: https://www.mdpi.com/2076-3417/11/15/7082

[^33]: https://www.sciencedirect.com/science/article/pii/S0034425721000845

[^34]: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JC017303

[^35]: https://www.tandfonline.com/doi/full/10.1080/15481603.2024.2393489

[^36]: https://modis.gsfc.nasa.gov/data/atbd/atbd_mod02.pdf

[^37]: https://www.sciencedirect.com/science/article/abs/pii/S0043135422010284

[^38]: https://www.tandfonline.com/doi/full/10.1080/15481603.2016.1177248

[^39]: https://www.sciencedirect.com/science/article/abs/pii/S0034425705004025

[^40]: https://www.sciencedirect.com/science/article/pii/S0048969725005388?dgcid=rss_sd_all

[^41]: https://citeseerx.ist.psu.edu/document?repid=rep1\&type=pdf\&doi=437dfa40508d04e2761a6d841409c159ddb42143

[^42]: https://www.mdpi.com/2072-4292/15/14/3534

[^43]: https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2017.00233/full

[^44]: https://ntrs.nasa.gov/api/citations/20210017930/downloads/JorgeDSF_2020_without_track_changes_4rd_round.docx.pdf

[^45]: https://www.researching.cn/ArticlePdf/m00032/2022/41/1/029.pdf


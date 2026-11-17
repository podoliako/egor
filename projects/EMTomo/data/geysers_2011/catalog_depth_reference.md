# Vertical reference of the LBNL / BG EGS catalogue

Research date: 2026-11-09.

## Conclusion and evidence level

Published analysis of the **same LBNL catalogue distributed through the NCEDC EGS catalogue search**, covering 2005–2014 (including our April 2011 interval), explicitly uses earthquake depths relative to **mean sea level**, not local ground surface.

This provides catalogue-specific literature support for interpreting our original `Depth` values as kilometres below mean sea level. It is stronger evidence than the depth convention in a subsequently relocated tomography catalogue. It is not a field-level specification from the catalogue operator: the NCEDC EGS help page itself does not state the vertical reference, and the paper does not explicitly discuss export-field transformations. The precise geodetic datum (e.g. NAVD88 versus another realization), station-elevation compatibility, and treatment of zero-depth solutions remain unverified.

No event coordinates, experiment metadata, or processing code were changed during this investigation.

## Our actual source

`events_2011_04.csv`, lines 1–12, records:

- `start_time=2011/04/01,00:00:00`
- `end_time=2011/04/30,23:59:59`
- `format=egscsv`
- `catalog=BG`
- event rows have `Source=LBL`.

This is the EGS/BG/LBNL catalogue, not the separate NCSN/NCSS catalogue or the older Calpine/Unocal UG archive.

## Main evidence: Johnson, Totten & Bürgmann (2016)

Christopher W. Johnson, Eoghan J. Totten, Roland Bürgmann,
*Depth migration of seasonally induced seismicity at The Geysers geothermal field*,
Geophysical Research Letters, 43(12).

- DOI: https://doi.org/10.1002/2016GL069546
- Repository: https://escholarship.org/uc/item/6x29k3d6
- Repository PDF: https://escholarship.org/content/qt6x29k3d6/qt6x29k3d6.pdf
- Full text successfully inspected through the PDF text proxy:
  https://r.jina.ai/https://escholarship.org/content/qt6x29k3d6/qt6x29k3d6.pdf
  (Direct repository/publisher access was blocked or returned an empty response during this search.)

### Section 2.1, Seismicity Catalog: exact source connection

> The seismicity catalog produced by LBNL is available from the Northern California Earthquake Data Center using the Enhanced Geothermal Systems Earthquake Catalog search (http://ncedc.org/egs/catalog-search.html, last accessed on 01 December 2015).

The same section specifies:

> The catalog is limited to the active production field (Figure 1) and contains 360,307 ML ≥ 0 events from 01 January 2005 to 31 December 2014, with 13 ML ≥ 4.0 events in the study area.

The authors analyse this catalogue; this is not a paper producing a new tomoDD-relocated catalogue. Section 2.1 describes the upstream hypocentral location procedure (SimulPS), rather than reporting an additional relocation performed for their study.

### Section 3, Seismicity Modeling: explicit vertical reference

> This procedure is performed for 1.0 km depth intervals from 1.0 km to 5.0 km with respect to mean sea level.

### Section 4: original catalogue used

> Figure 3 shows the monthly average injection and production totals, the cumulative number of M ≥ 1.5 events from the original catalog for every 0.5 km depth interval by 0.5 months, and the percent excess number of events in each bin for the NW and SE regions.

Together these statements connect the EGS/LBNL source catalogue and its depth analysis to mean sea level. The study covers our year, but its magnitude-filtered event selection and 2015 archive snapshot are not an exact copy of our downloaded April 2011 selection.

## Supporting evidence: Gritto, Yoo & Jarpe (2013)

*Three-Dimensional Seismic Tomography at The Geysers Geothermal Field, CA, USA*,
38th Stanford Workshop on Geothermal Reservoir Engineering, SGP-TR-198.

- PDF: https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2013/Gritto.pdf
- Text inspected through:
  https://r.jina.ai/https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2013/Gritto.pdf

Section **Seismic Data** identifies the LBNL network, automatically processed events archived at NCEDC, and approximately 32,000 selected events from 2011.

Section **Results and Discussion**:

> Vertical node spacing was also 600 m covering a depth range from -1 km to 4.5 km msl.

This is consistent with a mean-sea-level reference for the associated tomography. On its own it would not establish the vertical reference of the original EGS export, because this study simultaneously inverts for velocity and event locations.

## Official archive pages checked

- https://ncedc.org/egs/ identifies BG (formerly 3B/4B/5B), LBNL responsibility, and EGS catalogue availability from 2003-04-22 through 2016-06-30.
- https://ncedc.org/egs/catalog-search-help.html describes the EGS query but does not explicitly define the depth datum.

Generic NCSS documentation, Hypoinverse export-format conventions, and sea-level depths in Guo & Thurber's relocated catalogue are not treated as standalone proof for the original EGS data.

## Consequences for our box

`real_experiment_v1/geometry.json` currently uses:

```text
z_event_m   = 1166 + 1000 * Depth_km
z_station_m = 1166 - elevation_m
```

For an event depth positive below mean sea level and a flat box top at +1166 m in a compatible height reference, this is the appropriate sign and offset: `Depth=0` maps to `z=1.166 km`.

The literature supports this interpretation; it does **not** establish that catalogue zero-depth solutions are physically located at sea level. They could reflect upstream solution constraints, fixed/default depths, or another catalogue-processing issue. Their cause needs separate investigation. Nor does the article establish the exact geodetic compatibility of station elevations and event depths.

The existing conservative datum caveat in the prepared experiment was intentionally left unchanged to preserve provenance; this note records the newly found evidence separately.

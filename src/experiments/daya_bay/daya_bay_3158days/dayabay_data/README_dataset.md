# Daya Bay analysis dataset

- [Summary](<#summary>)
- [Citation statement](<#citation-statement>)
- [Files included](<#files-included>)
- [File types](<#file-types>)
    * [Naming and storage conventions](<#naming-and-storage-conventions>)
    * [Precision](<#precision>)
- [Input data](<#input-data>)
    * [IBD related data](<#ibd-related-data>)
        + [Prompt IBD spectra](<#prompt-ibd-spectra>)
        + [Detector performance data](<#detector-performance-data>)
        + [Background rates table](<#background-rates-table>)
        + [Shapes of the background spectra](<#shapes-of-the-background-spectra>)
    * [Common analysis inputs](<#common-analysis-inputs>)
        + [Detector performance](<#detector-performance>)
            - [IAV correction](<#iav-correction>)
            - [Liquid scintillator non-linearity (LSNL) correction](<#liquid-scintillator-non-linearity-lsnl-correction>)
        + [Neutrino rate data](<#neutrino-rate-data>)
            - [Weekly neutrino rate data](<#weekly-neutrino-rate-data>)
            - [Antineutrino spectra](<#antineutrino-spectra>)
            - [Non-equilibrium corrections](<#non-equilibrium-corrections>)
            - [Spent nuclear fuel (SNF) contribution as correction to the nominal spectrum](<#spent-nuclear-fuel-snf-contribution-as-correction-to-the-nominal-spectrum>)
        + [Input parameters](<#input-parameters>)
            - [Background rates](<#background-rates>)
            - [Detector related](<#detector-related>)
            - [Reactor related](<#reactor-related>)
            - [Electron antineutrino survival probability](<#electron-antineutrino-survival-probability>)
            - [Constants](<#constants>)
            - [Bin definitions](<#bin-definitions>)
            - [Conversion factors](<#conversion-factors>)

## Summary

The repository contains the full Daya Bay data set of inverse-beta-decay (IBD) candidates (reactor electron antineutrino interactions) with the final-state neutron captured on gadolinium. The dataset and supplementary data are sufficient to reproduce the measurement of neutrino oscillation parameters sin²2θ₁₃ and Δm²₃₂, published in [Phys.Rev.Lett. 130 (2023) 16, 161802](https://doi.org/10.1103/PhysRevLett.130.161802).

The Daya Bay Reactor Neutrino Experiment took data from 2011 to 2020 in China. It obtained a sample of 5.55 million IBD events with the final-state neutron captured on gadolinium (nGd). This sample was collected by eight identically designed antineutrino detectors (AD) observing antineutrino flux from six nuclear power plants located at baselines between 400 m and 2 km. It covers 3158 days of operation.

The analysis dataset is a light version of the full dataset, obtained by removing the tables with IBD events.

## Citation statement

If you use the dataset, cite the following sources:

[1] [Daya Bay Collaboration, “Full Data Release of the Daya Bay Reactor Neutrino Experiment”, v1.0.0. Zenodo, DOI:10.5281/zenodo.17587229; 2025](https://doi.org/10.5281/zenodo.17587229).

[2] [F. P. An et al. (Daya Bay collaboration), “Precision Measurement of Reactor Antineutrino Oscillation at Kilometer-Scale Baselines by Daya Bay”, Phys. Rev. Lett. 130,161802 (2023), DOI: 10.1103/PhysRevLett.130.161802](https://doi.org/10.1103/PhysRevLett.130.161802).

## Files included

- Detector performance data:
    * Daily data:
        + efficiency.
        + livetime.
        + rate of accidental background events.
    * Normalized background spectra.
    * Efficiencies and energy scale.
    * Background rates.
- Neutrino flux data:
    * weekly total neutrino rate from each reactor.
    * baselines relative to each detector.
- Reactor antineutrino data:
    * antineutrino spectra and corrections.
    * energy per fission, average fission fractions.
- Others:
    * all necessary physical constants.

## File types

There are 4 packages defined depending on the major data file type used: `npz`, `hdf5`, `tsv.bz2` (compressed text files) and `root` ([ROOT](https://root.cern.ch)). Each package also contains `yaml` files describing the model parameters and their uncertainties, and a few `python` files introducing physical constants. The contents and format of each data item is described below.

To run the analysis, only a single package is sufficient.

### Naming and storage conventions

There is a dedicated analysis code, which is able to read any of the provided packages. To support this across all the formats, a scheme is used on how to name the files and their contents. The data is typically contained in files (folders), containing (key, object) pairs. The objects are:

- Histograms:
    * (left edge, right edge, height) arrays.
    * `TH1D` objects: `root`.
- Graphs:
    * (x, y) arrays.
    * `root`:
        + `TH1D`. NOTE: In this case, x coordinate is stored as *left bin edge*.
        + `TGraph`, located in `graph/` subdirectory.
- Tables:
    * numpy records: `npz`, `hdf5`.
    * columns in a file with a header: `tsv.bz2`.
    * `TTree`: `root`.

The choice of `TH1D` histogram to store graphs is driven by accessibility: [uproot](https://github.com/scikit-hep/uproot5) package provides an easy way to read ROOT histograms, however does not support `TGraph` objects directly.

For the file naming conventions, the usual procedure is to request a module to load an object `key` from the file `filename`. The following rules are used:

- `npz` file `filename`.npz provides a dictionary with `{key: NDArray}`.
- `hdf5` file `filename`.hdf5 provides datasets (`NDArray`) with `key` used as dataset `name`.
- `root` file `filename`.root files contain (`TKey`, `TObject`) pairs.
- `tsv` files are more tricky. As it is not convenient to write multiple arrays into a text files, the corresponding to `npz`/`hdf5`/`root` file object is typically a folder with files. The following naming conventions are used:
    * `{filename}_{key}.tsv.bz2` — useful when only one array is stored, it is assumed that it is better to have a file with key in its name rather than a folder with a single file (`key`).
    * `{filename}.tsv/{key}.tsv.bz2` — `filename`.tsv is used as a folder with files, which names are keys.
    * `{filename}.tsv/{filename}_{key}.tsv.bz2` — similar layout as the previous one, however the filename is repeated to make nested files' names more verbose.

### Precision

Typically, "binary" data comes in double precision, the corresponding text files are printed with "%.17g" format, which should be sufficient to provide the same precision. Very abundant IBD data are provided in a single precision and the corresponding text files are printed with "%.9g" format.

## Input data

In this section, we will describe the contents of each file.

Time dependent data apart from a timestamp typically has and column `day` which denotes the 0-based serial number of a day since the start of data taking with 0 meaning December, 24 2011.

### IBD related data

The Day Bay experiment has three distinct periods of operation defined based on the number of detector being online: `6AD`, `8AD` and `7AD` for 6, 8 and 7 antineutrino detectors (AD) respectively. First period `6AD` had 6 detectors: 2 at first experimental hall EH1 (EH1AD1 or `AD11`, EH1AD2 or `AD12`), 1 at second experimental hall EH2 (`AD21`) and 3 at third (far) experimental hall EH3 (`AD31`, `AD32`, `AD33`). For second period `8AD` two more detectors were added: second detector `AD22` at EH2 and fourth detector `AD34` at EH3. In the last period, `7AD` `AD11` was not used for the physical data taking.

#### Prompt IBD spectra

- `hdf5/dayabay_dataset/` — the files with prompt IBD spectra for each data taking period `dayabay_ibd_spectra_6AD.hdf5`, `dayabay_ibd_spectra_7AD.hdf5`, `dayabay_ibd_spectra_8AD.hdf5` and for the full data taking period `dayabay_ibd_spectra_total.hdf5`.

Each file contains histograms with names of the form "ibd\_spectrum\_{detector}", e.g. `ibd_spectrum_AD11`. Energies are in MeV.

The files are obtained from the IBD events and a script to build them is also provided.

#### Detector performance data

- `hdf5/dayabay_dataset/dayabay_daily_detector_data.hdf5` — contains tables, stored by detector names (`AD11`, etc.)

Each table (dataset, record) contains the following columns:

- `start_utc` — start (midnight) time of the data taking day in seconds, UTC.
- `end_utc` — end (midnight) time of the data taking day in seconds, UTC.
- `start_date` — string representation of `start_utc`.
- `end_date` — string representation of `end_utc` end date and time
- `day` — serial number of a day since the start of data taking.
- `n_days` — number of days considered (1).
- `n_det` — number of active neutrino detectors.
- `livetime` — live time in seconds.
- `eff` — combined efficiency due to muon veto and multiplicity cuts.
- `rate_accidentals` — rate of accidental events (before efficiency applied).
- `eff_livetime` — effective live time (`livetime`\*`eff`).

#### Background rates table

- `hdf5/dayabay_dataset/dayabay_background_rates.hdf5` — tables with rates of backgrounds events for each detector for each period, stored by period names (`6AD`, `8AD`, and `7AD`).

Each table (dataset, record) contains the following:
- Columns:
    * `Label` — the label of the background and data type (rate or its uncertainty).
    * `AD11`, `AD12`, ... — detector names.
- Rows:
    * Rates per day:
        + `accidentals_rate` — for accidental coincidences.
        + `alpha_neutron_rate` — C(α,n)O reaction.
        + `amc_rate` — from AmC from automated calibration unit.
        + `fast_neutrons_rate` — fast neutrons. For `7AD` period includes also muon decay.
        + `lithium_helium_rate` — ⁹Li/⁸He related events.
    * Absolute uncertainties:
        + `accidentals_uncertainty` — for accidental coincidences.
        + `alpha_neutron_uncertainty` — C(α,n)O reaction.
        + `amc_uncertainty` — from AmC from automated calibration unit.
        + `fast_neutrons_uncertainty` — fast neutrons. For `7AD` period includes also muon decay.
        + `lithium_helium_uncertainty` — ⁹Li/⁸He related events.

Note, the information in this table is the same as information in [Background rates](<#background-rates>) yaml file. The rate of accidental events is consistent to the daily rate of accidentals in [detector performance data](<#detector-performance-data>).

#### Shapes of the background spectra

- `hdf5/dayabay_dataset/` — expected background event spectra for each period `dayabay_background_spectra_6AD.hdf5`, `dayabay_background_spectra_7AD.hdf5`, `dayabay_background_spectra_8AD.hdf5`.

Each file contains histograms for each source of backgrounds and each detector with names of format "spectrum\_shape\_{background\_name}\_{detector}". Histograms are normalized to 1. The backgrounds include:

- `accidentals` — for accidental coincidences.
- `alpha_neutron` — C(α,n)O reaction.
- `amc` — from AmC from automated calibration unit.
- `fast_neutrons` — fast neutrons. For `7AD` period includes also muon decay.
- `lithium_helium` — ⁹Li/⁸He related events.

### Common analysis inputs

#### Detector performance

##### IAV correction

The file `hdf5/detector_iav_matrix.hdf5` contains matrix `iav_correction`, which implements the distortion of the spectrum (240 equal bins from 0 to 12 MeV). The matrix is normalized so the sum of each column is 1.

##### Liquid scintillator non-linearity (LSNL) correction

The file `hdf5/detector_lsnl_curves.hdf5` contains 5 relative f(E) curves: the nominal energy non-linearity curve `nominal` and 4 pull curves (`pull0`, `pull1`, `pull2`, `pull3`) implementing systematic uncertainties.

#### Neutrino rate data

##### Weekly neutrino rate data

- `hdf5/neutrino_rate.hdf5` — weekly averaged neutrino rate tables for 6 sources (nuclear reactors): `R1`, `R2`, `R3`, `R4`, `R5`, `R6`.

Each table (dataset, record) contains the following columns:

- `period` — 0-based sequential number of aggregated period.
- `day` — sequential number of data taking day, corresponding to the start of the period.
- `start_utc` — start of the day (midnight) of the start of the period in seconds, UTC.
- `end_utc` — end of the day (midnight) of the end of the period in seconds, UTC.
- `n_days` — number of days in the period.
- `n_det` — number of active antineutrino detectors, e.g. corresponding data taking period.
- `n_det_mask` — binary mask indicating whether the period covers 6, 8 and 7 detector periods: `0b001` for 6 detectors, `0b010` for 8 detectors and `0b100` for 7 detectors.
- `neutrino_rate` — neutrino rate.

The neutrino rate is computed based on neutrino per fission for Huber-Mueller antineutrino spectra for exact IBD threshold and with exponential interpolation. The corresponding numbers of antineutrinos released per fission are provided in `neutrino_per_fission.yaml` (see below).

##### Antineutrino spectra

- `hdf5/reactor_antineutrino_spectra_hm.hdf5` — Huber-Mueller antineutrino spectra: graphs `U235`, `U238`, `Pu239`, `Pu241`. The spectra for ²³⁵U, ²³⁹Pu and ²⁴¹Pu from Huber et al., while ²³⁸U is from Mueller et al. The spectra are exponentially interpolated and extrapolated on a mesh of 50 keV.
- `hdf5/reactor_antineutrino_spectra_hm_uncertainties.hdf5` — corresponding correlated and uncorrelated antineutrino spectra uncertainties, interpolated and scaled for the mesh of 50 keV step.

References:

- [Phys.Rev.C 84 (2011) 024617](https://doi.org/10.1103/PhysRevC.85.029901), [Phys.Rev.C 85 (2012) 029901 (erratum)](https://doi.org/10.1103/PhysRevC.84.024617).
- [Phys.Rev.C 83 (2011) 054615](https://doi.org/10.1103/PhysRevC.83.054615).

##### Non-equilibrium corrections

- `hdf5/nonequilibrium_correction.hdf5` — relative non-equilibrium corrections to antineutrino spectra from Mueller et al.: graphs `U235`, `Pu239`, `Pu241`.

References:

- [Phys.Rev.C 83 (2011) 054615](https://doi.org/10.1103/PhysRevC.83.054615).

##### Spent nuclear fuel (SNF) contribution as correction to the nominal spectrum

- `hdf5/snf_correction.hdf5` — relative SNF related corrections (graphs) to nominal (with no non-equilibrium correction) antineutrino spectra from 6 reactors.

#### Input parameters

The folder `hdf5/parameters/` contains readable yaml (and python) files with text descriptions included. The files contain values of the parameters and their uncertainties. The type of the uncertainty is specified in the `format` field and may be:

- `sigma` — absolute uncertainty.
- `sigma_relative` — relative uncertainty.
- `sigma_percent` — relative uncertainty in %.

The Daya Bay model is designed to define all the parameters via configuration files, including constants and conversion factors. Therefore, the list contains all the parameters, including seemingly unimportant.

##### Background rates

The uncertainties of the background rates are stored in multiple files, as they have different correlation conditions and are loaded with different options (different number of clones is created). For the exact explanations on how to handle them, please, check the model in [5](https://github.com/dagflow-team/dayabay-model-official).

- `background_rate_scale_accidentals.yaml` — normalization uncertainty of the scale of the rate of accidentals, uncorrelated between detectors.
- `background_rate_uncertainty_scale_amc.yaml` — correlated between all the detectors uncertainty of the normalization (offset from 1) for the AmC background.
- `background_rate_uncertainty_scale_site.yaml` — correlated between detectors at the same experimental hall uncertainty of the normalization (offset from 1) for ⁹Li/⁸He and fast neutrons (and muon decay.)
- `background_rates_correlated.yaml` — background rates for the AmC, ⁹Li/⁸He, fast neutrons (and muon decay) backgrounds.
- `background_rates_uncorrelated.yaml` — background rates and uncertainties for Alpha-n background.

##### Detector related

- `baselines.yaml` — baseline distances between each reactor-detector pair, in meters.
- `detector_efficiency.yaml` — detector efficiency.
- `detector_eres.yaml` — detector energy resolution parameters and their uncertainties.
- `detector_iav_offdiag_scale.yaml` — scale correction and its uncertainty for the IAV correction.
- `detector_lsnl.yaml` — uncertainties of the LSNL model.
- `detector_normalization.yaml` — free detector normalization.
- `detector_n_protons_nominal.yaml` — nominal number of target protons in the detector.
- `detector_n_protons_correction.yaml` — corrections to the numbers of protons for each detector.
- `detector_relative.yaml` — partially correlated uncertainties for detector efficiency and energy scale factor.
- `extra/detector_absolute.yaml` — extra uncorrelated uncertainty for detector efficiency. Disabled by default.

##### Reactor related

- `neutrino_per_fission.yaml` — average number of neutrinos above the threshold released per fission for Huber+Mueller antineutrino spectra. These numbers were used to provide the neutrino rate per reactor.
- `reactor_energy_per_fission.yaml` — average energy released per fission for each isotope and its uncertainty.
- `reactor_fission_fractions.yaml` — average fission fractions for the whole data taking period for all the reactors.
- `reactor_fission_fractions_scale.yaml` — fission fractions' scale uncertainty for all isotopes and the correlation matrix for them.
- `reactor_nonequilibrium_correction.yaml` — uncertainty of the scale of non-equilibrium correction.
- `reactor_snf.yaml` — uncertainty of the scale of SNF related correction.
- `reactor_thermal_power_nominal.yaml` — nominal thermal power for reactors.
- `reactor_thermal_power_uncertainty.yaml` — uncertainty of the thermal power.

##### Electron antineutrino survival probability

- `survival_probability_constants.yaml` — definition of the neutrino mass ordering for the analysis.
- `survival_probability_solar.yaml` — solar neutrino oscillation parameters (sin²2θ₁₂ and Δm²₂₁) and their uncertainties.
- `survival_probability.yaml` — target neutrino oscillation parameters (sin²2θ₁₃ and Δm²₃₂), studied by Daya Bay.

##### Constants

- `ibd_constants.yaml` — constants for the inverse beta decay (IBD) cross-section.
- `pdg2024.yaml` — particle parameters from PDG2024.

##### Bin definitions

- `reactor_antineutrino_spectrum_edges.tsv` — definition of the mesh for the parameterization of average reactor antineutrino spectrum. A free parameter of the fit is added for each knot of the mesh.
- `final_erec_bin_edges.tsv` — definition of the final bins for the observation/prediction for each detector.

##### Conversion factors

There are two files `scipy` based calculation for conversion constants. They are detached from the other code so the high precision numbers are provided together with all the parameters.

- `conversion_survival_probability_argument.py` — Convert Δm²·L/E from [eV²·km/MeV] to natural units.
- `conversion_thermal_power.py` — Convert thermal power [GW/MeV]→[s⁻¹].


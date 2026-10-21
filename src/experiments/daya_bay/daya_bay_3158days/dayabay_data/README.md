# Full Data Release of the Daya Bay Reactor Neutrino Experiment

## Summary

The archive contains the full Daya Bay data set of inverse-beta-decay (IBD) candidates (reactor electron antineutrino interactions) with the final-state neutron captured on gadolinium. The dataset and supplementary data are sufficient to reproduce the measurement of neutrino oscillation parameters sin²2θ₁₃ and Δm²₃₂, published in [Phys.Rev.Lett. 130 (2023) 16, 161802](https://doi.org/10.1103/PhysRevLett.130.161802).

The Daya Bay Reactor Neutrino Experiment took data from 2011 to 2020 in China. It obtained a sample of 5.55 million IBD events with the final-state neutron captured on gadolinium (nGd). This sample was collected by eight identically designed antineutrino detectors (AD) observing antineutrino flux from six nuclear power plants located at baselines between 400 m and 2 km. It covers 3158 days of operation.

Code is provided elsewhere to read the dataset and produce a measurement of sin²2θ₁₃ and Δm²₃₂, consistent with the publication.

## Citation statement

If you use the dataset, cite the following sources:

[1] [Daya Bay Collaboration, “Full Data Release of the Daya Bay Reactor Neutrino Experiment”, v1.0.0. Zenodo, DOI:10.5281/zenodo.17587229; 2025](https://doi.org/10.5281/zenodo.17587229).

[2] [F. P. An et al. (Daya Bay collaboration), “Precision Measurement of Reactor Antineutrino Oscillation at Kilometer-Scale Baselines by Daya Bay”, Phys. Rev. Lett. 130,161802 (2023), DOI: 10.1103/PhysRevLett.130.161802](https://doi.org/10.1103/PhysRevLett.130.161802).

## The dataset organization

The data is provided in four different formats and is split into two categories:

1. Full dataset: IBD events in eight ADs, daily livetimes, daily efficiencies, rates of accidental backgrounds, and all the necessary inputs, needed to perform a measurement of sin²2θ₁₃ and Δm²₃₂. The size of the dataset is approximately 200 MB for each format.
2. Analysis dataset: all the necessary inputs, needed to perform a measurement of sin²2θ₁₃ and Δm²₃₂, including livetimes and efficiencies. The dataset includes IBD histograms for each data taking period. Its size is around 1 MB for each format.

Each category is available in four different formats: hdf5, npz, root and tsv (plain text, compressed). The detailed information on contents of the files and formats is provided in each archive.

The detailed description of the files and their contents may be found in [README_dataset.md](README_dataset.md).

## Acknowledgements

The results published here are in whole or in part based on the data released as Open-Access by the Daya Bay Collaboration supported by the Ministry of Science and Technology of China, the U.S. Department of Energy, the Chinese Academy of Sciences, the CAS Center for Excellence in Particle Physics, the National Natural Science Foundation of China, the New Cornerstone Science Foundation, the Guangdong provincial government, the Shenzhen municipal government, the China General Nuclear Power Group, the Research Grants Council of the Hong Kong Special Administrative Region of China, the National Science and Technology Council and the Ministry of Education in Taiwan, the U.S. National Science Foundation, the Ministry of Education, Youth, and Sports of the Czech Republic, the Charles University Research Centre UNCE, and the Joint Institute of Nuclear Research in Dubna, Russia.

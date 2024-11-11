This repository contains the code and instructions to reproduce the analysis from our paper, currently in revision for [Melba](https://www.melba-journal.org/)

Peoples, J.J., Hamghalam, M., James, I., Wasim, M., Gangai, N., Kang, H.C., Rong, X.J., Chun, Y.S., Do, R.K.G. and Simpson, A.L. (2024) “Finding Reproducible and Prognostic Radiomic Features in Variable Slice Thickness Contrast Enhanced CT of Colorectal Liver Metastases,” Machine Learning for Biomedical Imaging.

The paper is an extension of the work originally published in UNSURE 2023, held in conjunction with MICCAI:

Peoples, J.J., Hamghalam, M., James, I., Wasim, M., Gangai, N., Kang, H.C., Rong, X.J., Chun, Y.S., Do, R.K.G. and Simpson, A.L. (2023) “Examining the effects of slice thickness on the reproducibility of CT radiomics for patients with colorectal liver metastases,” in Uncertainty for Safe Utilization of Machine Learning in Medical Imaging. UNSURE 2023, pp. 42–52. https://doi.org/10.1007/978-3-031-44336-7_5.


# Preparing the data sets
## Extracting features
Features were extracted with pyradiomics using the settings files in `pyradiomics_settings/`.

## Preparing the reproducibility data set
The reproducibility imaging data set is currently not publicly available, (although we plan to release it in the future). The structure of the imaging is

For every patient, we have 21 reconstructions of a single scan

- 7 levels of ASiR {0,10,20,30,40,50,60}
- 3 slice thicknesses {2.5, 3.75, 5}

For each patient we have a segmentation of the scan (in 5mm slice thickness coordinates), which delineates the liver parenchyma, vessels, and tumor regions.

To prepare the tables, features were extracted with pyradiomics from all 21 images for each patient, using the same masks (1 for liver parenchyma, 1 for largest tumor), which were interpolated to each reconstruction (when needed) using nearest neighbor interpolation.

## Preparing the public survival data set
The data are available [here](https://doi.org/10.7937/QXK2-QG03).

The DICOM data can be converted to Niftii using the instructions [here](https://github.com/jpeoples/tcia_crlm_data_conversion)

To obtain the liver parenchyma: subtract all other masks from the Liver mask for each patient.

To obtain the largest tumor: take the tumor mask with the largest number of non-zero voxels.

# Reproducing the analysis

## Pairwise CCC analysis
To compute the pairwise CCC analysis (Fig 3), run

```
python melba.py pairwise_cccs
```

To produce the corresponding figure, run

```
python melba.py fig_pairwise_cccs
```

## LMM-based CCC analysis
To compute the LMM-based CCCs, run

```
python melba.py lmm_cccs
```

This will call the R script `ccc_lmm_auto.R` in order to compute the lmm-based CCC for every feature.

# Univariable survival vs reproducibility analysis

### Univariable C-indexes
To compute the table of univariate C-indexes, run

```
python melba.py univariate_survival
```

### Cluster map figures
To produce the cluster map figures (in Fig 4):

```
python melba.py cluster_figs
```

### Pareto front figures
To produce the Pareto front plots, as well as the other ranking plots in Fig 5, and the LaTeX source for table 5:

```
python melba.py pareto_figs
```

## Multivariable analysis
Because this section entails running 100 times repeated 10-fold cross validation across 315 configurations of feature set, CCC threshold, and feature selection count, (i.e. 315000 models are fitted), this section was designed to run on a SLURM cluster in batches.

To run a batch:

```
python melba.py multivariate_survival --feature_set <feature-set-index> --feature_count <feature-count-index> --feature_selection <feature-selection-index> --jobs <no.-jobs>
```

This will run the 100 times repeated 10-fold cross val for the selected experiment. Each parameter is an index into the corresponding list of configurations given in conf.json.

The resulting tables must be combined into a single large file using `join_outputs.sh`.

Once all batches are executed, to produce the corresponding figures and tables (Table 4, Fig 6, Fig 7)

```
python melba.py multivariate_figs
```

## Additional analyses
To produce the list of all features in Tab 6, and the list of feature counts in Tab 1

```
python melba.py generate_additional_tables
```

To produce various statistics used in the discussion/results section:

```
python melba.py final_statistics
```

This computes the range of p-values when comparing the C-indexes of features from the liver vs tumor, the frequency of features selected from the liver ROI in the top two models, etc

To produce the visualizations of features with different configurations of high/low CCC and high/low C-index (Fig 8): 

```
python melba.py generate_feature_visualizations
```

To generate the histograms in Figs 1 and 2,

```
python melba.py image_spacing_histograms
```

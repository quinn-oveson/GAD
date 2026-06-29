# UAV-Based Spatiotemporal Phenotyping and Growth Modeling for Forecasting Potato Yield

[Access this dataset on Dryad](https://doi.org/10.5061/dryad.5qfttdzmn)

This dataset provides the data and reproducible code used for pre-harvest yield forecasting of potato using UAV-based remote sensing. Field experiments were conducted in 2023 and 2024. For each experimental plot, RGB and multispectral UAV imagery were acquired, and vegetation features were extracted from orthomosaic images and digital surface models (DSM). In addition, destructive sampling was conducted at multiple time points to measure tuber fresh weight (TW).

Using these UAV-derived features and observed TW data, a Random Forest regression model was developed to estimate TW at different harvest stages. Furthermore, Gompertz growth curves were fitted to the time-series of estimated TW in order to forecast yield at the expected harvest date.

This repository includes plot-level ground-truth TW datasets (.csv), UAV-derived feature datasets (.csv), and analysis scripts provided as R Markdown files (.Rmd) and Jupyter Notebook files (.ipynb).

## Description of the file structure
The ZIP file named **Potato_UAV_Yield_Forecasting.zip** contains all data files and analysis scripts (R Markdown and Jupyter Notebook files) used in this study.
The original folder structure has been preserved to ensure reproducibility and clarity of the workflow.
A description of the file organization is provided below.

```text
Potato_UAV_Yield_Forecasting/
│
├── README.md
│
├── 01_data/
│   ├── FT_2023_df.csv
│   ├── FT_2024_df.csv
│   ├── TW_2023_df.csv
│   ├── TW_2024_df.csv
│   └── 0_model_timestamp_config.csv
│
└── 02_code/
    ├── 011_2023_plot_ground_truth_TW.Rmd
    ├── 012_2024_plot_ground_truth_TW.Rmd
    ├── 021_2023_TW_Estimation.ipynb
    ├── 022_2024_TW_Estimation.ipynb
    ├── 031_2023_TW_Forecasting.Rmd
    └── 032_2024_TW_Forecasting.Rmd
```

## Data
Contains plot-level datasets used for model training and evaluation.

### TW_YYYY_df.csv 
This file contains plot-level observed tuber fresh weight (TW) data obtained through destructive sampling. Each row corresponds to a single experimental plot and includes treatment information, days after sowing (DAS) at destructive sampling, and TW (g m⁻²). These data were used for training and evaluating TW estimation models based on UAV-derived features.

* Plot_ID: Unique identifier assigned to each plot  
* Origin: Seed tuber origin (experimental factor in 2023)  
* Temperature: Seed tuber temperature treatment (experimental factor in 2024)  
* Density: Planting density treatment  
* DAS: Days After Sowing  
* Wei_all_per_m2: Tuber fresh weight per unit area (g m⁻²)

### FT_YYYY_df.csv
This file contains plot-level UAV-derived feature data extracted from RGB and multispectral imagery. Each row corresponds to a single experimental plot and includes treatment information and the target variable (Wei_all_per_m2, TW), together with vegetation features derived from UAV imagery.

Feature columns follow a prefix-based naming convention (e.g., PH_, CC_, NDVI_), with acquisition dates appended as suffixes. The analysis scripts select feature columns based on these prefixes to construct the dataset used for model training and evaluation.

* block: Block identifier used for block-wise cross-validation (block-wise CV) to spatially separate training and validation data.
* PH_*: Canopy height (90th percentile above baseline)
* CC_*: Canopy cover (% area)
* PV_*: Canopy volume (height × area)
* ExG_*: Excess Green Index
* VARI_*: Visible Atmospherically Resistant Index
* GRVI_*: Green–Red Vegetation Index
* NDVI_*: Normalized Difference Vegetation Index
* GNDVI_*: Green Normalized Difference Vegetation Index
* NDRE_*: Normalized Difference Red Edge

Note: Plot_ID, Origin, Temperature, and Density are described in section TW_YYYY_df.csv.  
Note: For detailed definitions of feature calculations, see Table 1 in the main manuscript.

### 0_model_timestamp_config.csv
This file is a configuration file used to manage model training timestamps. When model training is executed, the execution date and time are automatically assigned as a label. In the visualization scripts, the corresponding timestamp must be specified in advance to identify which trained model results to reference. This file defines the mapping between timestamps, cross-validation (CV) methods, and experimental years.

**Example structure**
| CV_name      | 2023         | 2024         |
|-------------|-------------|-------------|
| RandomCV    | "YYYYMMDDHHmm"| "YYYYMMDDHHmm"|
| BlockwiseCV | "YYYYMMDDHHmm"| "YYYYMMDDHHmm"|

Note: Values represent model training timestamps in the format "YYYYMMDDHHmm".

## Software/Code
### Software Environment
All analyses were conducted using:
* R version: 4.5.1
    * dplyr: 1.1.4
    * ggplot2: 4.0.1
    * nlme: 3.1.168

* Python 3.8.19
    * numpy: 1.24.3
    * pandas: 2.0.3
    * scikit-learn: 1.3.0
    * matplotlib: 3.7.2

### 01*_YYYY_plot_ground_truth_TW.Rmd
This script visualizes observed TW data collected in each year. It reads *`TW_YYYY_df.csv`* and plots the mean TW of each treatment against DAS. It is primarily used to examine the distribution and temporal trends of the observed data.

### 02*_YYYY_TW_Estimation.ipynb
This script estimates TW using UAV-derived features. It reads *`FT_YYYY_df.csv`* as input data and uses TW at each harvest time as the target variable to train a machine learning model (Random Forest). Model performance is evaluated using cross-validation, including both random CV and block-wise CV. When appropriate, feature selection and preprocessing steps are incorporated into a pipeline, and optimal hyperparameters are identified using GridSearchCV. This script performs training and evaluation of the TW estimation model and saves the resulting predictions and evaluation metrics.

### 02*_plot_YYYY_TW_Estimation.ipynb
This script visualizes the results of TW estimation model. It reads the predicted values and evaluation metrics generated by *`02*_YYYY_TW_Estimation.ipynb`* and plots the results of feature selection as well as the relationship between observed and predicted TW. This script is primarily used to compare model performance.

### 03*_YYYY_TW_Forecasting.Rmd
This script forecasts yield at harvest using the time-series of estimated TW. It takes the estimated TW obtained from *`02*_YYYY_TW_Estimation.ipynb`*  as input and models the growth process by fitting a Gompertz growth curve. Based on the fitted growth curves, TW at the expected harvest date is predicted, and forecasting accuracy is evaluated by comparison with observed values. This script represents the final stage of the yield forecasting framework based on UAV-derived features.

## Acknowledgement
ChatGPT (OpenAI) was used as an assistive tool in the preparation and organization of this README and the associated code. It was also used to improve the clarity and readability of the documentation. The authors reviewed and edited the content as needed and take full responsibility for the content of this repository.

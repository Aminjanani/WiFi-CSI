# Wi-Fi CSI People Counting

An embedded system that uses Wi-Fi Channel State Information (CSI)
captured through a wireless router to estimate the number of people in
an indoor environment.

## Project structure

``` text
CSI_People_Counting/
├── README.md
├── requirements.txt
├── firmware/
│   └── esp32_csi_receiver/
│       └── esp32_csi_receiver.ino
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── metadata/
│   ├── scenarios.csv
│   └── sessions.csv
├── src/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── modeling/
│   └── utils/
├── notebooks/
├── models/
└── reports/
    ├── figures/
    └── evaluation/
```

## Directory overview

  -----------------------------------------------------------------------
  Path                                Purpose
  ----------------------------------- -----------------------------------
  `firmware/`                         ESP32 firmware for connecting to
                                      the Wi-Fi router, receiving CSI
                                      data, and transmitting records over
                                      USB Serial.

  `data/raw/`                         Original serial logs and raw CSI
                                      records. Keep these files
                                      unchanged.

  `data/interim/`                     Parsed and organized intermediate
                                      data, such as packet-level CSV
                                      files and window indexes.

  `data/processed/`                   Cleaned and transformed data ready
                                      for feature extraction or model
                                      training.

  `metadata/`                         Scenario labels and
                                      recording-session information,
                                      including moving and stationary
                                      person counts.

  `src/data_collection/`              Scripts for capturing serial output
                                      and saving each recording session.

  `src/preprocessing/`                Scripts for parsing CSI records,
                                      checking data quality, and creating
                                      time windows.

  `src/feature_extraction/`           Code for deriving features from CSI
                                      measurements.

  `src/modeling/`                     Model training, evaluation, and
                                      inference code.

  `src/utils/`                        Shared configuration and helper
                                      functions.

  `notebooks/`                        Exploratory analysis, experiments,
                                      visualizations, and
                                      model-development notebooks.

  `models/`                           Trained model files and related
                                      artifacts.

  `reports/figures/`                  Figures and plots generated during
                                      analysis and evaluation.

  `reports/evaluation/`               Evaluation results, metrics, and
                                      experiment summaries.

  `requirements.txt`                  Python dependencies required by the
                                      project.
  -----------------------------------------------------------------------

## Data collection notes

-   Store each independent recording in its own session directory under
    `data/raw/`.
-   Preserve the original serial log so that data can be reparsed later.
-   Record scenario labels and session details in `metadata/`.
-   Keep target labels (for example, `moving_count` and
    `stationary_count`) separate from CSI-derived input features.
-   Split training and test data by recording session rather than
    randomly splitting overlapping time windows, to reduce data leakage.

## Initial workflow

1.  Capture CSI packets from the ESP32 and save the raw serial log.
2.  Parse the log into packet-level records.
3.  Check data quality and create time windows.
4.  Extract features from the CSI data.
5.  Train and evaluate people-counting models.
6.  Save trained models and document the results.

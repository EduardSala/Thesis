
# My Master of Sciences thesis 🎓


## Context 🌊

 The Offshore Renewable Energy sector continuously requires accurate data, particularly
 regarding significant wave height and wind speed, which are key variables necessary
 to characterize the operational and environmental conditions of offshore wind farms.
 
 In this context, accurate data can support decision-making for a specific site to meet
 these conditions and, most importantly, facilitate continuous energy generation. In-situ
 platforms can help collect accurate data, but they are quite expensive and frequently
 experience malfunctions, leading to discontinuous observations.
 
 In this scenario, satellites play an important role as they are capable of acquiring accurate
 data with discrete temporal resolution.
 

## Goals 🎯

 This thesis aims to evaluate the performance of satellite altimetry data by collocating it
 with fixed-point positions, between satellite altimetry data and in-situ data, using a spatio
 temporal matching method, exploring different criterions.
 
 Moreover, several bias correction techniques are applied to calibrate satellite data against
 in-situ data to improve the quality of satellite assimilated dataset, aligning it more closely
 with in-situ and and unlock their potential in providing environmental insights.

## Workflow ⚡

```mermaid
graph TD
    %% Input
    SAT[Satellite Altimetry Data] --> START(( ))
    INS[In-Situ Mooring Data] --> START
    START --> SPACE{Spatial Matching}

    %% Sezione Spaziale
    SPACE --> R30[Radius: 30 km]
    SPACE --> R50[Radius: 50 km]
    SPACE --> R70[Radius: 70 km]
    
    R30 & R50 & R70 --> S_METHODS[<b>Spatial Co-location Methods</b>]

    %% Sezione Temporale
    S_METHODS --> TIME{Temporal Matching}
    
    TIME --> T15[Window: 15 min]
    TIME --> T30[Window: 30 min]
    TIME --> T60[Window: 60 min]
    
    T15 & T30 & T60 --> T_METHODS[<b>Temporal Co-location Methods</b>]

    %% Sezione Bias Correction
    T_METHODS --> BC_START{Bias Correction Techniques}
    
    BC_START --> BC1[Full Distribution Mapping]
    BC_START --> BC2[Quantile Mapping]
    BC_START --> BC3[Linear Regression]
    BC_START --> BC4[Delta Technique]

    %% Sezione Validazione (Metriche)
    BC1 & BC2 & BC3 & BC4 --> VAL[<b>Data Validation</b>]
    
    VAL --> RMSE[RMSE]
    VAL --> BIAS_M[BIAS]
    VAL --> CC[CC]
    VAL --> SI[SI]

    %% Risultato Finale
    RMSE & BIAS_M & CC & SI --> FINAL[/Final Calibrated Dataset/]
```

## Repository Structure 📂

- [`Module_all_functions.py`](./Module_all_functions.py): Core library containing all shared functions used across the project.
- [`Spatio_temporal.py`](./Spatio_temporal.py): Implementation of the spatio-temporal matching algorithm and co-location logic.
- [`calibration.py`](./calibration.py): Script for applying the 4 bias correction techniques and generating comparative plots for the 10 datasets.


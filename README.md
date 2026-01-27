
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
    SAT[Satellite Altimetry Data]
    INS[In-Situ Mooring Data]

    %% Flusso Satellite (Spazio)
    subgraph ST [ ]
        direction TB
        L1[<b>1a. Spatial Co-location</b>]
        SAT --> R[Radii: 30, 50, 70 km]
        R --> S_MET[Methods: Minimum Distance, IDW]
    end

    %% Flusso Mooring (Tempo)
    subgraph TT [ ]
        direction TB
        L2[<b>1b. Temporal Co-location</b>]
        INS --> W[Windows: 15, 30, 60 min]
        W --> T_MET[Methods: Closest Obs., Mean Value]
    end

    %% Punto di Sincronizzazione
    S_MET --> SYNC([Spatio-Temporal Synchronization])
    T_MET --> SYNC

    %% Processing
    subgraph CP [ ]
        direction TB
        L3[<b>2. Calibration </b>]
        SYNC --> BC{Bias Correction Techniques}
        BC --> BC1[Full Distribution Mapping]
        BC --> BC2[Quantile Mapping]
        BC --> BC3[Linear Regression]
        BC --> BC4[Delta Technique]
    end

    %% Output e Validazione
    BC1 & BC2 & BC3 & BC4 --> SAVE[(Storage of all scenarios)]
    
    SAVE --> COMP{Comparison Analysis}

    subgraph VR [ ]
        direction TB
        L4[<b>3. Validation & Performance</b>]
        COMP --> METRICS["Statistical Parameters<br/>(RMSE, BIAS, CC, SI)"]
    end

    METRICS --> FINAL[/Identification of the most accurate configuration/]
```

## Repository Structure 📂

- [`Module_all_functions.py`](./Module_all_functions.py): Core library containing all shared functions used across the project.
- [`Spatio_temporal.py`](./Spatio_temporal.py): Implementation of the spatio-temporal matching algorithm and co-location logic.
- [`calibration.py`](./calibration.py): Script for applying the 4 bias correction techniques and generating comparative plots for the 10 datasets.


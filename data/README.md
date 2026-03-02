# 🗄️ Datasets selection
In this section I explain why and which datasets have been used in this calibration workflow.

## ⚓ In-situ data (moorings)

In-situ data are often used as a reference value, for accuracy purposes, for the calibration and quality validation of the satellite altimetry data. Most commonly in-situ data is identified as _ground truth_ because can be seen that they deliver the most accurate data, compared to other instruments. For this work, mooring platforms have been used as the main source of in-situ data delivering.

In the last two decades the number of platforms that deliver data such as wind speed, wave height, sea level an so on, has grown exponentially. In this scenario there are multiple providers that give access to all type of in-situ data, such as U.S National Data Buoy Center (_NDBC_) or Copernicus Service (_CMEMS_). For this work all the mooring platforms have been selected on the In-Situ TAC, which is is the component of the Copernicus Marine Service  guarantee a reliable access to a wide range of in-situ data. Furthermore, all the data are Near Real Time Observations (_NRT_) because we want to compare those data to the satellite altimetry data.

Lastly, all the mooring distancing more than 100 km have been selected so that satellite measurements when selected from more than a 50 km radius, are not influenced by noise near the coast. 

<p align="center">
<img src="https://github.com/user-attachments/assets/cc9ffc5f-6da1-440c-ba7b-4b2e30f207dc" width="320">
<img src="https://github.com/user-attachments/assets/13dfaf59-3a6a-4611-bdf2-fe338f8431e5" width="320">
<br>
<em>From left to right: all the 200+ mooring buoys have been filtered to 18.</em>
</p>

| Platform | Start date | End date | Latitude | Longitude | Dcoast [km] | D. closest platf. [km] | Platform |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Gullfaks-C | 01/01/2014 | On-going | 2.26 | 61.20 | 124 | 19.94 | Visundfeltet |
| Oseberg-A | 05/01/2014 | On-going | 2.82 | 60.49 | 105 | 11.37 | Oseberg-SOR |
| Oseberg-SOR | 02/06/2014 | On-going | 2.79 | 60.39 | 109 | 11.37 | Oseberg-A |
| 6300112* | 19/05/2011 | On-going | 1.0 | 61.1 | 100 | 49 | Statfjord-A |
| Snorre-A | 06/01/2014 | On-going | 2.14 | 61.44 | 131 | 9.20 | Snorre-B |
| Snorre-B | 06/01/2014 | On-going | 2.20 | 61.52 | 129 | 9.20 | Snorre-A |
| Statfjord-A | 01/01/1978 | On-going | 1.85 | 61.25 | 146 | 23.08 | Gullfaks-C |
| Visundfeltet | 24/04/2014 | On-going | 2.43 | 61.36 | 115 | 18.11 | Snorre-A |
| A122* | 06/07/2015 | On-going | 3.81 | 55.41 | 241 | 85.53 | F3platform |
| F3platform* | 06/01/2014 | On-going | 4.72 | 54.85 | 164 | 85.53 | A122 |
| J61* | 19/05/2011 | On-going | 2.95 | 53.81 | 142 | 69.01 | K13a3 |
| K13a3* | 19/05/2011 | On-going | 3.22 | 53.21 | 101 | 69.01 | J61 |
| Ekofisk | 11/01/1980 | On-going | 3.22 | 56.54 | 262 | 100.23 | 6200146 |
| Sleipner-A | 01/01/2014 | On-going | 1.90 | 58.37 | 199 | 50.99 | 6200130 |
| Granefeltet | 09/04/2014 | On-going | 2.48 | 59.16 | 135 | 67.44 | 6300110 |
| 6200130* | 07/07/2014 | On-going | 1.3 | 58.7 | 190 | 50.99 | Sleipner-A |
| 6200146* | 07/07/2014 | On-going | 2.1 | 57.2 | 234 | 100.23 | Ekofisk |
| 6300110* | 19/05/2011 | On-going | 1.5 | 59.5 | 157 | 67.44 | Granefeltet |

(* _those with the asterisk are the ones that do not deliver **wind speed data**_)
### 🗄️ [Access to In-situ Dataset](http://data.marine.copernicus.eu/product/INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030/description)

To be precise, all the platform data has been downloaded from the [In-situ TAC Dashboard](https://marineinsitu.eu/dashboard/), where the GUI made it possible to access data in easier way.

<p align="center">
<img src="https://github.com/user-attachments/assets/a835adbb-de20-4b8a-bc15-e6b33d1ba050" width="1020">
<br>
<em>In-situ TAC Dashboard.</em>
</p>



---

## 🛰️ Satellite altimetry data

For the satellite data, the following dataset has been selected from the same provider as before, CMEMS. This product is based on NRT measurements with processing level L3 of significant wave height and wind speed. The following tables summarize the main information about the entire dataset and the temporal availability for each satellite mission. More details can be found on the product page in the Copernicus data store.

### 📝 Description

<div align="center">

| | |
| :--- | :--- |
| **Full name** | Global Ocean L 3 Significant Wave Height From Nrt Satellite Measurements |
| **Product ID** | WAVE_GLO_PHY_SWH_L3_NRT_014_001 |
| **Variables** | Significant Wave Height and Wind Speed at 10-m |
| **Spatial extent** | Global Ocean |
| **Missions** | Sentinel-6A; Jason-3; Sentinel-3A; Sentinel-3B; SAR-AL/AltiKa; Cryosat2; CFOSAT; HaiYang-2B; HaiYang-2C, SWOT nadir |
| **Spatial resolution** | Along-track ~ 7 km (full 1 Hz resolution) |
| **Temporal resolution** | Instantaneous |

</div>

### 🕓 Temporal availability

<div align="center">

| Satellite | Begin date | End date | Cross-over points* |
| :---: | :---: | :---: | :---: |
| Altika | 01/01/2021 | on-going | 2231 |
| C2 | 01/01/2021 | on-going | 1920 |
| Cfosat | 01/01/2021 | on-going | 2615 |
| H2b | 01/01/2021 | on-going | 2142 |
| H2c | 01/12/2022 | on-going | 1623 |
| J3 | 01/01/2021 | on-going | 3407 |
| S3a | 01/01/2021 | on-going | 2556 |
| S3b | 01/01/2021 | on-going | 2757 |
| S6a | 21/09/2021 | on-going | 2632 |
| Swon | 01/08/2023 | on-going | 680 |
<br>
<em>Temporal availability of the different satellite missions. *Cross-over points have been calculated using a 50 km cross-radius.</em>

</div>

### 🗄️ [Access to Altimetry Dataset](https://data.marine.copernicus.eu/product/WAVE_GLO_PHY_SWH_L3_NRT_014_001/description)




















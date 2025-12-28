
# Airline Departure Delay Prediction Dataset

## Dataset Source
**Repository:** Mendeley Data  
**Dataset Type:** Research-grade tabular dataset  
**Domain:** Aviation analytics, time-series prediction, weather-aware machine learning  

---

## Dataset Overview
This dataset contains commercial airline flight records augmented with airport-specific weather observations and temporal congestion indicators. It is designed to support predictive modeling of flight departure and arrival delays using pre-departure operational and meteorological conditions.

The dataset integrates:
- Airline operational data (scheduling, carriers, routes)
- Airport congestion context
- Weather data from authoritative meteorological sources (e.g., Iowa Environmental Mesonet)

Each row represents one scheduled flight, with features capturing conditions before and around the scheduled departure time.

---

## Temporal Coverage
- Calendar year: **2023**
- Daily flight operations
- Hour-based temporal windows around departure and arrival

---

## Intended Use
- Flight delay classification (e.g., delay > 15 minutes)
- Weather impact analysis on airline punctuality
- Airport congestion and delay propagation modeling
- Explainable and operationally realistic ML systems

---

## Feature Descriptions

### 1. Calendar & Temporal Features
- MONTH: Month of flight operation  
- DAY_OF_MONTH: Day of the month  
- FL_DATE: Flight date  
- day_of_week: Day of week  

### 2. Airline & Route Information
- MKT_CARRIER: Marketing airline code  
- OP_CARRIER: Operating airline code  
- ORIGIN: Origin airport code  
- DEST: Destination airport code  
- DISTANCE: Flight distance (miles)  
- FAA_class: Aircraft size/class  

### 3. Scheduled Time & Duration (Pre-departure)
- CRS_ELAPSED_TIME: Scheduled flight duration  
- Scheduled_DEP: Scheduled departure time  
- Scheduled_ARR_Ori: Scheduled arrival (origin timezone)  
- Scheduled_DEP_EST: Scheduled departure (EST)  
- Scheduled_ARR_EST: Scheduled arrival (EST)  
- Scheduled_ARR_Local: Scheduled arrival (local time)  

### 4. Airport Congestion & Time-Window Features
- CRS_DEP_1hrpre: Scheduled departure window 1 hour before  
- CRS_DEP_1hrpost: Scheduled departure window 1 hour after  
- DEP_1hrpre_num: Departures 1 hour before  
- DEP_1hrpost_num: Departures 1 hour after  
- Arr_1hrpre_num: Arrivals 1 hour before  
- Arr_1hrpost_num: Arrivals 1 hour after  

### 5. Weather Features
- max_temp_f: Maximum temperature (°F)  
- min_temp_f: Minimum temperature (°F)  
- max_dewpoint_f: Maximum dew point  
- min_dewpoint_f: Minimum dew point  
- precip_in: Precipitation (inches)  
- snow_in: Snowfall (inches)  
- avg_wind_speed_kts: Average wind speed (knots)  
- avg_feel: Feels-like temperature  

### 6. Turnaround & Operational Context
- scheduled_Turnarnd: Scheduled turnaround time  
- Actual_Turnarnd: Actual turnaround time (post-event)  
- Diff_in_turnarnd: Difference between scheduled & actual  
- longTurnaround: Indicator of extended turnaround  

### 7. Delay Target Variables (Labels)
- DEP_DELAY: Departure delay (minutes)  
- DEP_DELAY_NEW: Non-negative departure delay  
- DEP_DEL15: Departure delay > 15 min (binary)  
- ARR_DELAY: Arrival delay (minutes)  
- ARR_DELAY_NEW: Non-negative arrival delay  
- ARR_DEL15: Arrival delay > 15 min (binary)  

### 8. Delay Cause Breakdown (Post-event)
- CARRIER_DELAY: Airline-caused delay  
- WEATHER_DELAY: Weather-caused delay  
- NAS_DELAY: Air traffic system delay  
- SECURITY_DELAY: Security-related delay  
- LATE_AIRCRAFT_DELAY: Previous aircraft delay  

---

## Important Modeling Note
This dataset contains both pre-departure predictive features and post-event outcome variables. Careful feature selection is required to prevent data leakage and ensure realistic predictive performance.

---

## One-Line Summary
A research-grade dataset combining U.S. commercial flight operations with airport-level weather and temporal congestion features to enable pre-departure flight delay prediction.

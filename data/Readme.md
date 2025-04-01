# BTS Flight Data Documentation

## Overview
This documentation describes the data extracted from the Bureau of Transportation Statistics (BTS) "On-Time Performance" database for the SkyDenied flight delay detection ML group project. The data contains comprehensive information about airline flights, with a focus on departure and arrival delays.

## Selected Features
The SkyDenied group project uses the following key features from the BTS dataset:

| Feature | Description |
|---------|-------------|
| `FlightDate` | Date of the flight (yyyymmdd format) |
| `Reporting_Airline` | Unique carrier code identifier |
| `Flight_Number_Reporting_Airline` | Flight number |
| `Tail_Number` | Aircraft tail number |
| `Origin` | Origin airport code |
| `Dest` | Destination airport code |
| `CRSDepTime` | Scheduled departure time (local time: hhmm) |
| `CRSArrTime` | Scheduled arrival time (local time: hhmm) |
| `DepDelayMinutes` | Departure delay in minutes (early departures set to 0) |
| `ArrDelayMinutes` | Arrival delay in minutes (early arrivals set to 0) |
| `Cancelled` | Flight cancellation indicator (1=Yes) |

## Model Information
- **Project Name**: SkyDenied
- **Purpose**: Flight delay detection and prediction
- **Target Variables**: `DepDelayMinutes` and `ArrDelayMinutes`
- **Input Features**: All other listed features

## Data Source
The data is extracted from the Reporting Carrier On-Time Performance (2022) data table of the "On-Time" database from the TranStats data library maintained by the Bureau of Transportation Statistics.

## Additional Notes
- Time fields are in local time using 24-hour format (hhmm)
- Delay values for early departures/arrivals are set to 0 in the `*DelayMinutes` fields
- The `Reporting_Airline` code is consistent across years and should be used for analysis instead of IATA codes

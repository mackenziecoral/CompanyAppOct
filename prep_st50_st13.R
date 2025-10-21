#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lubridate)
})

normalize_facid <- function(x) {
  x <- ifelse(is.na(x), "", toupper(trimws(as.character(x))))
  gsub("\\s+", "", x)
}

st13_path <- "data/st13b_2025_detail.csv"
st50_path <- "data/st50_gas_plant_master.csv"

if (!file.exists(st13_path)) {
  stop("Missing ST13 source: ", st13_path)
}
if (!file.exists(st50_path)) {
  stop("Missing ST50 source: ", st50_path)
}

st13 <- read_csv(st13_path, guess_max = 1e6, show_col_types = FALSE) %>%
  rename(
    gas_throughput_km3 = `Gas Plant Disposition (1000 cu.m.)`,
    lat = `Facility Latitude`,
    lon = `Facility Longitude`,
    plant_type = `Facility Type`,
    sub_type = `Facility Sub Type Short Description`,
    operator = `Facility Operator BA Name`,
    facid = `Facility ID`
  ) %>%
  mutate(
    facid = normalize_facid(facid),
    ym = ym(sprintf("%04d-%02d", Year, Month)),
    days = days_in_month(ym),
    # ST13 disposition is monthly 10^3 m³; convert to average daily 10^3 m³/d
    throughput_km3_per_d = gas_throughput_km3 / pmax(days, 1)
  )

st50 <- read_csv(st50_path, show_col_types = FALSE) %>%
  mutate(`Licensed Inlet Capacity (1000 m3/d)` = coalesce(
    `Licensed Inlet Capacity (1000 m3/d)`,
    `Licensed Inlet Capacity (mmscfd)` * 28.174
  )) %>%
  rename(
    cap_km3_per_d = `Licensed Inlet Capacity (1000 m3/d)`,
    lat_cap = `Facility Latitude`,
    lon_cap = `Facility Longitude`,
    operator_cap = `Facility Operator BA Name`,
    plant_type_cap = `Facility Type`,
    facid = `Facility ID`,
    FacilityName = `Facility Name`
  ) %>%
  mutate(facid = normalize_facid(facid))

prepared <- st13 %>%
  left_join(select(st50, facid, FacilityName, cap_km3_per_d, lat_cap, lon_cap, operator_cap, plant_type_cap), by = "facid")

write_csv(prepared, "data/st_gas_plants_prepped.csv")

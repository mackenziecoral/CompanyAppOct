# --- 0. Load Necessary Packages ---
library(shiny)
library(sf)
library(leaflet)
library(dplyr)
library(readxl)
library(DT)
library(tidyr)
library(stringr)
library(shinyWidgets)
library(RColorBrewer)
library(ggplot2)
library(scales)
library(data.table)
library(lwgeom)
library(tools)
library(htmltools)
library(viridis)
library(leaflet.extras)
library(DBI)
library(odbc)
library(glue)
library(lubridate)
library(plotly)
# library(minpack.lm) # For Arps if needed, currently nls is used

# --- Database Connection Details ---
db_user <- "WOODMAC"
db_password <- "c0pp3r"
tns_alias <- "GDC_LINK.geologic.com"
oracle_driver <- "Oracle in instantclient_23_7" # Adjust if your driver name is different

connection_string <- paste0(
  "Driver={", oracle_driver, "};",
  "Dbq=", tns_alias, ";",
  "Uid=", db_user, ";",
  "Pwd=", db_password, ";"
)

con <- NULL

# --- Function to Establish Database Connection ---
connect_to_db <- function() {
  if (!is.null(con) && dbIsValid(con)) {
    # If connection exists and is valid, no need to disconnect and reconnect, just return it.
    # However, if this function is explicitly called to force a new connection,
    # then we might want to disconnect first. For a general "get connection" this is fine.
    return(con)
  }
  local_con <- NULL
  tryCatch({
    message("Attempting to connect to Oracle: ", tns_alias, " as user: ", db_user)
    Sys.sleep(0.1)
    local_con <- dbConnect(
      odbc::odbc(),
      .connection_string = connection_string,
      timeout = 10
    )
    message("SUCCESS: Database connection established.")
    return(local_con)
  }, error = function(e) {
    message("ERROR during dbConnect call: Failed to connect to Oracle.")
    message("Detailed R error: ", e$message)
    return(NULL)
  })
}

# Initial connection attempt
con <- connect_to_db()

onStop(function() {
  if (!is.null(con) && dbIsValid(con)) {
    dbDisconnect(con)
    message("Disconnected from Oracle database on app stop.")
  }
})

# --- 1. Define File Paths and Constants ---
base_path <- "C:/Users/I37643/OneDrive - Wood Mackenzie Limited/Documents/WoodMac/APP" # UPDATE THIS PATH
processed_rds_file <- file.path(base_path, "processed_app_data_vMERGED_FINAL_v24_DB_sticks_latlen.rds")

woodmack_coverage_file_xlsx <- file.path(base_path, "Woodmack.Coverage.2024.xlsx")
play_subplay_shapefile_dir <- file.path(base_path, "SubplayShapefile")
company_shapefiles_dir <- file.path(base_path, "Shapefile")


# Conversion factors
E3M3_TO_MCF <- 35.3147
M3_TO_BBL <- 6.28981
AVG_DAYS_PER_MONTH <- 30.4375
MCF_PER_BOE <- 6 # Standard conversion for BOE calculations

# Custom Color Palette
custom_palette <- c(
  "#53143F", "#F94355", "#058F96", "#F57D01", "#205B2E", "#A8011E",
  "#5A63E3", "#FFD31A", "#4E207F", "#CC9900", "#A6A6A6", "#9ACEE2",
  "#A9899E", "#FFA3AA", "#92C5C9", "#F9BE96", "#92AC96", "#D6890B",
  "#ABAFF0", "#FFE89C", "#A68EBF", "#D7C68E", "#D1D1D1", "#CBE5EF"
)
if (!all(sapply(custom_palette, function(x) grepl("^#[0-9A-Fa-f]{6}$", x) || x %in% colors()))) {
  warning("Some colors in custom_palette may not be valid R colors or hex codes. Using fallback for invalid ones.")
  custom_palette <- sapply(custom_palette, function(x) {
    if (grepl("^#[0-9A-Fa-f]{6}$", x) || x %in% colors()) return(x)
    if (grepl("^#[0-9A-Fa-f]{5}$", x) && nchar(x) == 6) return(paste0(x, "B"))
    return("grey")
  })
}


# --- 2. Helper Functions ---
standardize_uwi <- function(uwi_vector) {
  uwi_vector <- as.character(uwi_vector)
  uwi_vector <- stringr::str_replace_all(uwi_vector, "[^A-Za-z0-9]", "")
  uwi_vector <- toupper(uwi_vector)
  return(uwi_vector)
}

safe_read_excel <- function(file_path, sheet_name, file_description = file_path) {
  message(paste("Attempting to load Excel sheet:", sheet_name, "from", file_description))
  if (!file.exists(file_path)) {
    warning(paste("Excel file not found:", file_path, "for", file_description))
    return(data.frame())
  }
  tryCatch({
    df <- readxl::read_excel(file_path, sheet = sheet_name, .name_repair = "universal")
    message(paste("Successfully loaded Excel sheet:", sheet_name, "- Rows:", nrow(df), "Cols:", ncol(df)))
    return(as.data.frame(df))
  }, error = function(e) {
    warning(paste("Error loading Excel sheet", sheet_name, "from", file_path, ": ", e$message))
    return(data.frame())
  })
}

clean_df_colnames <- function(df_input, df_name_for_message = "a dataframe") {
  if (is.null(df_input) || (!is.data.frame(df_input) && !is.data.table(df_input)) || ncol(df_input) == 0) {
    return(df_input)
  }
  df <- if (is.data.table(df_input)) data.table::copy(df_input) else df_input
  original_names <- names(df)
  new_names <- make.names(original_names, unique = TRUE)
  new_names <- gsub("\\.+", "_", new_names)
  new_names <- gsub("_+", "_", new_names)
  new_names <- toupper(new_names)
  if (is.data.table(df)) {
    data.table::setnames(df, original_names, new_names)
  } else {
    names(df) <- new_names
  }
  return(df)
}

prepare_filter_choices <- function(column_vector, col_name_for_msg = "column") {
  if (is.null(column_vector) || length(column_vector) == 0) {
    message(paste0("DEBUG (prepare_filter_choices): Input vector for '", col_name_for_msg, "' is NULL or empty."))
    return(character(0))
  }
  column_vector_char <- as.character(column_vector)
  num_na_initial <- sum(is.na(column_vector_char))
  num_empty_initial <- sum(column_vector_char == "", na.rm = TRUE)
  
  choices <- column_vector_char[!is.na(column_vector_char) & column_vector_char != "" & column_vector_char != "NA"]
  choices <- sort(unique(choices))
  
  message(paste0("DEBUG (prepare_filter_choices for '", col_name_for_msg, "'): ",
                 "Original NAs: ", num_na_initial, ", Original empty strings: ", num_empty_initial,
                 ". Final unique choices: ", length(choices)))
  if(length(choices) == 0) return(character(0))
  return(choices)
}

load_process_spatial_layer <- function(shp_path, layer_name, target_crs = 4326, simplify = FALSE, tolerance = NULL, make_valid_geom = FALSE) {
  message(paste0("--- Start Processing Layer: ", layer_name, " ---"))
  message(paste0("Shapefile path: ", shp_path))
  tryCatch({
    if (!file.exists(shp_path)) {
      warning(paste("Shapefile does not exist at path:", shp_path, "for layer", layer_name)); return(NULL)
    }
    data_sf <- sf::st_read(shp_path, quiet = TRUE, stringsAsFactors = FALSE)
    if (nrow(data_sf) == 0) {
      warning(paste("Shapefile for layer", layer_name, "is empty.")); return(NULL)
    }
    message(paste("Layer", layer_name, "read successfully, rows:", nrow(data_sf)))
    current_crs <- sf::st_crs(data_sf)
    if (is.na(current_crs) || (is.list(current_crs) && is.na(current_crs$epsg)) || (is.list(current_crs) && !is.na(current_crs$epsg) && current_crs$epsg != target_crs)) {
      message(paste("Transforming layer", layer_name, "to CRS", target_crs))
      data_sf <- sf::st_transform(data_sf, crs = target_crs)
    }
    if (make_valid_geom) {
      message(paste("Making geometries valid for layer", layer_name))
      data_sf <- sf::st_make_valid(data_sf)
    }
    if (simplify && !is.null(tolerance) && tolerance > 0) {
      message(paste("Simplifying layer", layer_name, "with tolerance", tolerance))
      data_sf <- sf::st_simplify(data_sf, dTolerance = tolerance, preserveTopology = TRUE)
    }
    
    data_sf <- data_sf[!sf::st_is_empty(data_sf), ]
    if(nrow(data_sf) == 0) {
      warning(paste("All geometries became empty for layer", layer_name, "after processing. Skipping."))
      return(NULL)
    }
    
    attrs_df <- sf::st_drop_geometry(data_sf)
    cleaned_attrs_df <- clean_df_colnames(attrs_df, paste("attributes of", layer_name))
    data_sf_final <- sf::st_sf(cleaned_attrs_df, geometry = sf::st_geometry(data_sf))
    if (!"SHP_LAYER_NAME" %in% names(data_sf_final)) data_sf_final$SHP_LAYER_NAME <- layer_name
    message(paste("--- Successfully processed layer:", layer_name, "- Final Features:", nrow(data_sf_final)))
    return(data_sf_final)
  }, error = function(e) {
    message(paste("!!!!!!!! ERROR processing shapefile", shp_path, "for layer", layer_name, ":", e$message, "!!!!!!!!"))
    return(NULL)
  })
}

# Custom geometric mean function
geometric_mean <- function(x, na.rm = TRUE) {
  if (na.rm) {
    x <- x[!is.na(x)]
  }
  if (any(x < 0) || length(x) == 0) {
    return(NA_real_)
  }
  if (any(x == 0)) {
    return(0)
  }
  exp(mean(log(x)))
}


# --- 3. Load and Pre-process Data ---
final_sf_column_names <- c(
  "UWI", "GSL_UWI", "SurfaceLatitude", "SurfaceLongitude",
  "BH_Latitude", "BH_Longitude", "LateralLength",
  "AbandonmentDate", "WellName", "CurrentStatus", "OperatorCode", "StratUnitID",
  "SpudDate", "FirstProdDate", "FinalTD", "ProvinceState", "Country",
  "UWI_Std", "GSL_UWI_Std", "OperatorName", "Formation", "FieldName",
  "ConfidentialType"
)
empty_wells_df_for_sf <- data.frame(matrix(ncol = length(final_sf_column_names), nrow = 0))
names(empty_wells_df_for_sf) <- final_sf_column_names
empty_wells_df_for_sf$UWI <- character(); empty_wells_df_for_sf$GSL_UWI <- character()
empty_wells_df_for_sf$SurfaceLatitude <- numeric(); empty_wells_df_for_sf$SurfaceLongitude <- numeric()
empty_wells_df_for_sf$BH_Latitude <- numeric(); empty_wells_df_for_sf$BH_Longitude <- numeric()
empty_wells_df_for_sf$LateralLength <- numeric()
empty_wells_df_for_sf$AbandonmentDate <- as.Date(character()); empty_wells_df_for_sf$WellName <- character()
empty_wells_df_for_sf$CurrentStatus <- character(); empty_wells_df_for_sf$OperatorCode <- character()
empty_wells_df_for_sf$StratUnitID <- character(); empty_wells_df_for_sf$SpudDate <- as.Date(character())
empty_wells_df_for_sf$FirstProdDate <- as.Date(character()); empty_wells_df_for_sf$FinalTD <- numeric()
empty_wells_df_for_sf$ProvinceState <- character(); empty_wells_df_for_sf$Country <- character()
empty_wells_df_for_sf$UWI_Std <- character(); empty_wells_df_for_sf$GSL_UWI_Std <- character()
empty_wells_df_for_sf$OperatorName <- character(); empty_wells_df_for_sf$Formation <- character()
empty_wells_df_for_sf$FieldName <- character()
empty_wells_df_for_sf$ConfidentialType <- character()

app_data <- list(
  wells_sf = sf::st_sf(empty_wells_df_for_sf, geometry = sf::st_sfc(), crs = 4326),
  play_subplay_layers_list = list(),
  company_layers_list = list()
)
load_from_db <- TRUE

if (file.exists(processed_rds_file)) {
  message(paste("Attempting to load MERGED pre-processed data from:", processed_rds_file))
  tryCatch({
    loaded_data <- readRDS(processed_rds_file)
    
    wells_ok <- !is.null(loaded_data$wells_sf) && inherits(loaded_data$wells_sf, "sf") &&
      nrow(loaded_data$wells_sf) > 0 &&
      all(final_sf_column_names %in% names(sf::st_drop_geometry(loaded_data$wells_sf))) &&
      ("CompletionDate" %in% names(loaded_data$wells_sf) && inherits(loaded_data$wells_sf$CompletionDate, "Date"))
    
    play_layers_ok <- !is.null(loaded_data$play_subplay_layers_list) && is.list(loaded_data$play_subplay_layers_list)
    company_layers_ok <- !is.null(loaded_data$company_layers_list) && is.list(loaded_data$company_layers_list)
    
    if (play_layers_ok && length(loaded_data$play_subplay_layers_list) > 0) {
      play_layers_ok <- all(sapply(loaded_data$play_subplay_layers_list, function(x) !is.null(x$data) && inherits(x$data, "sf")))
    }
    if (company_layers_ok && length(loaded_data$company_layers_list) > 0) {
      company_layers_ok <- all(sapply(loaded_data$company_layers_list, function(x) !is.null(x$data) && inherits(x$data, "sf")))
    }
    
    if (wells_ok && play_layers_ok && company_layers_ok) {
      app_data <- loaded_data
      message("SUCCESS: Pre-processed data (including wells and layer lists) loaded and validated from RDS.")
      load_from_db <- FALSE
    } else {
      message("WARNING: RDS file loaded, but wells_sf data or layer lists are invalid/incomplete/empty. Will load from DB and re-process layers.")
      app_data$wells_sf = sf::st_sf(empty_wells_df_for_sf, geometry = sf::st_sfc(), crs = 4326)
      app_data$play_subplay_layers_list <- list()
      app_data$company_layers_list <- list()
      load_from_db <- TRUE
    }
  }, error = function(e) {
    message(paste("ERROR loading RDS file:", processed_rds_file, "-", e$message))
    message("Will proceed to load data from DB and shapefiles.")
    app_data$wells_sf = sf::st_sf(empty_wells_df_for_sf, geometry = sf::st_sfc(), crs = 4326)
    app_data$play_subplay_layers_list <- list()
    app_data$company_layers_list <- list()
    load_from_db <- TRUE
  })
} else {
  message("Pre-processed RDS file not found. Will load from DB and shapefiles.")
}

message(paste0("--- Status before potential DB/Shapefile load: load_from_db = ", load_from_db, " ---"))

if (load_from_db) {
  message("INITIATING DATA LOAD FROM DATABASE AND/OR SHAPEFILES...")
  if(is.null(app_data$wells_sf) || nrow(app_data$wells_sf) == 0 || !all(final_sf_column_names %in% names(app_data$wells_sf))) {
    app_data$wells_sf <- sf::st_sf(empty_wells_df_for_sf, geometry = sf::st_sfc(), crs = 4326)
  }
  app_data$play_subplay_layers_list <- list()
  app_data$company_layers_list <- list()
  
  if (is.null(con) || !dbIsValid(con)) { message("Attempting to (re)connect to database for data loading..."); con <- connect_to_db(); if (is.null(con) || !dbIsValid(con)) { stop("FATAL: Database connection failed. Cannot load primary data.") } }
  
  sql_well_master_base <- paste0(
    "SELECT W.UWI, W.GSL_UWI, W.SURFACE_LATITUDE, W.SURFACE_LONGITUDE, ",
    "W.BOTTOM_HOLE_LATITUDE, W.BOTTOM_HOLE_LONGITUDE, W.GSL_FULL_LATERAL_LENGTH, ",
    "W.ABANDONMENT_DATE, W.WELL_NAME, W.CURRENT_STATUS, W.OPERATOR AS OPERATOR_CODE, W.CONFIDENTIAL_TYPE, ",
    "P.STRAT_UNIT_ID, W.SPUD_DATE, PFS.FIRST_PROD_DATE, W.FINAL_TD, W.PROVINCE_STATE, W.COUNTRY, FL.FIELD_NAME ",
    "FROM WELL W ",
    "LEFT JOIN PDEN P ON W.GSL_UWI = P.GSL_UWI ",
    "LEFT JOIN FIELD FL ON W.ASSIGNED_FIELD = FL.FIELD_ID ",
    "LEFT JOIN PDEN_FIRST_SUM PFS ON W.GSL_UWI = PFS.GSL_UWI ",
    "WHERE W.SURFACE_LATITUDE IS NOT NULL AND W.SURFACE_LONGITUDE IS NOT NULL ",
    "AND (W.ABANDONMENT_DATE IS NULL OR W.ABANDONMENT_DATE > SYSDATE - (365*20))"
  )
  message("Fetching well master data from Oracle..."); wells_master_df_raw <- tryCatch({ dbGetQuery(con, sql_well_master_base) }, error = function(e) { warning(paste("Error fetching well master data from Oracle:", e$message)); data.frame() })
  wells_master_dt <- data.table::data.table()
  if (nrow(wells_master_df_raw) > 0) {
    message(paste("DB Load: Successfully loaded", nrow(wells_master_df_raw), "base well rows from DB.")); wells_master_dt <- data.table::as.data.table(wells_master_df_raw)
    if("UWI" %in% names(wells_master_dt)) wells_master_dt[, UWI_Std := standardize_uwi(UWI)] else wells_master_dt[, UWI_Std := NA_character_]
    if("GSL_UWI" %in% names(wells_master_dt)) wells_master_dt[, GSL_UWI_Std := standardize_uwi(GSL_UWI)] else wells_master_dt[, GSL_UWI_Std := NA_character_]
    if (!"FIELD_NAME" %in% names(wells_master_dt)) { wells_master_dt[, FieldName := NA_character_] } else { setnames(wells_master_dt, "FIELD_NAME", "FieldName") }
    if (!"STRAT_UNIT_ID" %in% names(wells_master_dt)) wells_master_dt[, STRAT_UNIT_ID := NA_character_]; wells_master_dt[, STRAT_UNIT_ID := as.character(STRAT_UNIT_ID)]
    
    if ("CONFIDENTIAL_TYPE" %in% names(wells_master_dt)) {
      wells_master_dt[, CONFIDENTIAL_TYPE := as.character(CONFIDENTIAL_TYPE)]
    } else {
      wells_master_dt[, CONFIDENTIAL_TYPE := NA_character_]
    }
    if (!"BOTTOM_HOLE_LATITUDE" %in% names(wells_master_dt)) wells_master_dt[, BOTTOM_HOLE_LATITUDE := NA_real_]
    if (!"BOTTOM_HOLE_LONGITUDE" %in% names(wells_master_dt)) wells_master_dt[, BOTTOM_HOLE_LONGITUDE := NA_real_]
    wells_master_dt[, BOTTOM_HOLE_LATITUDE := as.numeric(BOTTOM_HOLE_LATITUDE)]
    wells_master_dt[, BOTTOM_HOLE_LONGITUDE := as.numeric(BOTTOM_HOLE_LONGITUDE)]
    if (!"GSL_FULL_LATERAL_LENGTH" %in% names(wells_master_dt)) wells_master_dt[, GSL_FULL_LATERAL_LENGTH := NA_real_]
    wells_master_dt[, GSL_FULL_LATERAL_LENGTH := as.numeric(GSL_FULL_LATERAL_LENGTH)]
    
    
    unique_strat_ids <- unique(wells_master_dt$STRAT_UNIT_ID[!is.na(wells_master_dt$STRAT_UNIT_ID) & wells_master_dt$STRAT_UNIT_ID != ""])
    if (length(unique_strat_ids) > 0) {
      batch_size <- 500; strat_names_list <- list()
      for (i in seq(1, length(unique_strat_ids), by = batch_size)) {
        batch_ids <- unique_strat_ids[i:min(i + batch_size - 1, length(unique_strat_ids))]
        sql_strat_unit_names <- glue::glue_sql("SELECT STRAT_UNIT_ID, SHORT_NAME FROM STRAT_UNIT WHERE STRAT_UNIT_ID IN ({ids*})", ids = batch_ids, .con = con)
        strat_names_batch_df <- tryCatch({ dbGetQuery(con, sql_strat_unit_names) }, error = function(e) { data.frame()}); if(nrow(strat_names_batch_df) > 0) strat_names_list[[length(strat_names_list) + 1]] <- strat_names_batch_df
      }
      if (length(strat_names_list) > 0) {
        strat_names_df <- rbindlist(strat_names_list, use.names = TRUE, fill = TRUE); strat_names_dt <- data.table::as.data.table(strat_names_df)
        setnames(strat_names_dt, "SHORT_NAME", "Formation", skip_absent=TRUE); if (!"Formation" %in% names(strat_names_dt)) strat_names_dt[, Formation := NA_character_]
        if ("STRAT_UNIT_ID" %in% names(strat_names_dt)) strat_names_dt[, STRAT_UNIT_ID := as.character(STRAT_UNIT_ID)] else strat_names_dt[, STRAT_UNIT_ID := NA_character_]
        if("STRAT_UNIT_ID" %in% names(wells_master_dt) && "STRAT_UNIT_ID" %in% names(strat_names_dt)){ wells_master_dt <- merge(wells_master_dt, strat_names_dt[, .(STRAT_UNIT_ID, Formation)], by = "STRAT_UNIT_ID", all.x = TRUE, sort = FALSE)
        } else { wells_master_dt[, Formation := NA_character_] }
      } else { wells_master_dt[, Formation := NA_character_] }
    } else { wells_master_dt[, Formation := NA_character_] }
  } else { message("WARNING (DB Load): No well master data returned from Oracle. wells_master_dt is empty.") }
  operator_codes_df_raw <- safe_read_excel(woodmack_coverage_file_xlsx, sheet_name = "Operator", "Woodmack Coverage (Sheet: Operator)")
  operator_codes_dt <- clean_df_colnames(data.table::as.data.table(operator_codes_df_raw), "Operator Codes from Excel"); operator_codes_final_dt <- data.table()
  if (nrow(operator_codes_dt) > 0 && all(c("OPERATOR", "GSL_PARENT_BA_NAME") %in% names(operator_codes_dt))) {
    operator_codes_final_dt <- operator_codes_dt[, .(WoodmackJoinOperatorCode = as.character(OPERATOR), OperatorNameDisplay = GSL_PARENT_BA_NAME)][!is.na(WoodmackJoinOperatorCode) & WoodmackJoinOperatorCode != "" & !is.na(OperatorNameDisplay) & OperatorNameDisplay != ""]; operator_codes_final_dt <- unique(operator_codes_final_dt, by = "WoodmackJoinOperatorCode")
  }
  combined_wells_dt <- data.table()
  if (nrow(wells_master_dt) > 0) {
    if ("OPERATOR_CODE" %in% names(wells_master_dt)) {
      wells_master_dt[, OPERATOR_CODE := as.character(OPERATOR_CODE)]
      if (nrow(operator_codes_final_dt) > 0) { combined_wells_dt <- merge(wells_master_dt, operator_codes_final_dt, by.x = "OPERATOR_CODE", by.y = "WoodmackJoinOperatorCode", all.x = TRUE)
      } else { combined_wells_dt <- wells_master_dt; combined_wells_dt[, OperatorNameDisplay := NA_character_] }
    } else { combined_wells_dt <- wells_master_dt; combined_wells_dt[, OperatorNameDisplay := NA_character_] }
  }
  if (nrow(combined_wells_dt) > 0) {
    db_to_r_names_map <- c(
      "UWI"="UWI", "GSL_UWI"="GSL_UWI", "SURFACE_LATITUDE"="SurfaceLatitude",
      "SURFACE_LONGITUDE"="SurfaceLongitude",
      "BOTTOM_HOLE_LATITUDE"="BH_Latitude", "BOTTOM_HOLE_LONGITUDE"="BH_Longitude",
      "GSL_FULL_LATERAL_LENGTH"="LateralLength", # Updated mapping for LateralLength
      "ABANDONMENT_DATE"="AbandonmentDate", "WELL_NAME"="WellName", "CURRENT_STATUS"="CurrentStatus",
      "OPERATOR_CODE"="OperatorCode", "STRAT_UNIT_ID"="StratUnitID",
      "SPUD_DATE"="SpudDate", "FIRST_PROD_DATE"="FirstProdDate",
      "FINAL_TD"="FinalTD", "PROVINCE_STATE"="ProvinceState", "COUNTRY"="Country",
      "UWI_Std"="UWI_Std", "GSL_UWI_Std"="GSL_UWI_Std",
      "OperatorNameDisplay"="OperatorName", "Formation"="Formation", "FieldName"="FieldName",
      "CONFIDENTIAL_TYPE"="ConfidentialType"
    )
    current_db_names <- names(combined_wells_dt)
    for (db_name in names(db_to_r_names_map)) {
      r_name <- db_to_r_names_map[[db_name]];
      if (db_name %in% current_db_names && db_name != r_name) {
        setnames(combined_wells_dt, db_name, r_name, skip_absent = TRUE)
      } else if (db_name %in% current_db_names && db_name == r_name && !r_name %in% current_db_names[current_db_names != db_name]){
        # Column already has the target R name, do nothing
      } else if (!r_name %in% names(combined_wells_dt) && db_name %in% current_db_names) {
        setnames(combined_wells_dt, db_name, r_name, skip_absent = TRUE)
      }
    }
    
    for(col_sf in final_sf_column_names){
      if(!col_sf %in% names(combined_wells_dt)) {
        col_type <- class(empty_wells_df_for_sf[[col_sf]]);
        if (col_type == "numeric") combined_wells_dt[, (col_sf) := NA_real_]
        else if (col_type == "Date") combined_wells_dt[, (col_sf) := as.Date(NA_character_)]
        else combined_wells_dt[, (col_sf) := NA_character_]
      } else {
        if (col_sf == "ConfidentialType" && !is.character(combined_wells_dt[[col_sf]])) {
          combined_wells_dt[, (col_sf) := as.character(get(col_sf))]
        }
        if (col_sf == "BH_Latitude" && !is.numeric(combined_wells_dt[[col_sf]])) {
          combined_wells_dt[, (col_sf) := as.numeric(get(col_sf))]
        }
        if (col_sf == "BH_Longitude" && !is.numeric(combined_wells_dt[[col_sf]])) {
          combined_wells_dt[, (col_sf) := as.numeric(get(col_sf))]
        }
        if (col_sf == "LateralLength" && !is.numeric(combined_wells_dt[[col_sf]])) {
          combined_wells_dt[, (col_sf) := as.numeric(get(col_sf))]
        }
      }
    }
    
    if("SurfaceLatitude" %in% names(combined_wells_dt) && !is.numeric(combined_wells_dt$SurfaceLatitude)) combined_wells_dt[, SurfaceLatitude := as.numeric(SurfaceLatitude)]
    if("SurfaceLongitude" %in% names(combined_wells_dt) && !is.numeric(combined_wells_dt$SurfaceLongitude)) combined_wells_dt[, SurfaceLongitude := as.numeric(SurfaceLongitude)]
    date_cols_to_convert_pascal <- c("SpudDate", "FirstProdDate", "AbandonmentDate")
    for(dc_pascal in date_cols_to_convert_pascal){ if(dc_pascal %in% names(combined_wells_dt) && !inherits(combined_wells_dt[[dc_pascal]], "Date")){ current_col_values <- combined_wells_dt[[dc_pascal]]; if(inherits(current_col_values, "POSIXct") || inherits(current_col_values, "POSIXlt")) { combined_wells_dt[, (dc_pascal) := as.Date(current_col_values)] } else { combined_wells_dt[, (dc_pascal) := as.Date(as.character(current_col_values), origin = "1970-01-01")] } } }
    combined_wells_for_sf <- combined_wells_dt[!is.na(SurfaceLatitude) & !is.na(SurfaceLongitude)]
    if (nrow(combined_wells_for_sf) > 0) {
      combined_wells_for_sf_df <- as.data.frame(combined_wells_for_sf[, ..final_sf_column_names]); coord_names_to_use <- c("SurfaceLongitude", "SurfaceLatitude")
      if (any(is.na(combined_wells_for_sf_df[[coord_names_to_use[1]]])) || any(is.na(combined_wells_for_sf_df[[coord_names_to_use[2]]]))) { combined_wells_for_sf_df <- combined_wells_for_sf_df[ !is.na(combined_wells_for_sf_df[[coord_names_to_use[1]]]) & !is.na(combined_wells_for_sf_df[[coord_names_to_use[2]]]), ] }
      if(nrow(combined_wells_for_sf_df) > 0) {
        app_data$wells_sf <- sf::st_as_sf(combined_wells_for_sf_df, coords = coord_names_to_use, crs = 4269, agr = "constant", remove = FALSE) %>% sf::st_transform(crs = 4326)
        if(any(!sf::st_is_valid(app_data$wells_sf), na.rm = TRUE)) app_data$wells_sf <- app_data$wells_sf[which(sf::st_is_valid(app_data$wells_sf) %in% TRUE), ]
      } else { message("DB Load: No valid rows after NA filter for coordinates before sf creation.")}
    } else { message("DB Load: No valid rows after filtering for non-NA PascalCase coordinates.") }
  } else { message("DB Load: combined_wells_dt is empty, no wells_sf created from DB.") }
  message(paste("DB Load: app_data$wells_sf object created/updated with", nrow(app_data$wells_sf), "features."))
  
  # --- SHAPEFILE LOADING LOGIC ---
  message("--- Loading Play/Subplay Acreage ---")
  message(paste("Checking directory:", play_subplay_shapefile_dir))
  play_subplay_layers_list_temp <- list()
  if (dir.exists(play_subplay_shapefile_dir)) {
    shp_files <- list.files(play_subplay_shapefile_dir, pattern = "\\.shp$", full.names = TRUE, ignore.case = TRUE, recursive = TRUE)
    message(paste("Found", length(shp_files), "potential play/subplay shapefiles in:", play_subplay_shapefile_dir))
    if(length(shp_files) > 0){
      play_subplay_layers_list_temp <- lapply(shp_files, function(p) {
        lname <- tools::file_path_sans_ext(basename(p))
        message(paste("Attempting to load play/subplay layer:", lname, "from:", p))
        ldata <- load_process_spatial_layer(p, lname, simplify = TRUE, tolerance = 100, make_valid_geom = TRUE)
        if (!is.null(ldata) && nrow(ldata) > 0 && inherits(ldata, "sf")) {
          message(paste("SUCCESS: Loaded play/subplay layer:", lname, "with", nrow(ldata), "features."))
          list(name = lname, data = ldata)
        } else {
          message(paste("FAILED or EMPTY: Play/subplay layer:", lname, "from:", p))
          NULL
        }
      })
      app_data$play_subplay_layers_list <- Filter(Negate(is.null), play_subplay_layers_list_temp)
    } else {
      message("No .shp files found in the play/subplay directory.")
    }
  } else { message(paste("Play/Subplay directory NOT FOUND:", play_subplay_shapefile_dir)) }
  message(paste("Loaded", length(app_data$play_subplay_layers_list), "valid play/subplay layers."))
  
  message("--- Loading DISSOLVED Company Acreage ---")
  message(paste("Checking directory:", company_shapefiles_dir))
  company_layers_list_temp <- list()
  if (dir.exists(company_shapefiles_dir)) {
    shp_files <- list.files(company_shapefiles_dir, pattern = "\\.shp$", full.names = TRUE, ignore.case = TRUE, recursive = FALSE)
    message(paste("Found", length(shp_files), "potential company shapefiles in:", company_shapefiles_dir))
    if(length(shp_files) > 0){
      company_layers_list_temp <- lapply(shp_files, function(p) {
        lname <- tools::file_path_sans_ext(basename(p))
        message(paste("Attempting to load company layer:", lname, "from:", p))
        ldata <- load_process_spatial_layer(p, lname, simplify = FALSE, make_valid_geom = FALSE)
        if (!is.null(ldata) && nrow(ldata) > 0 && inherits(ldata, "sf")) {
          message(paste("SUCCESS: Loaded company layer:", lname, "with", nrow(ldata), "features."))
          list(name = lname, data = ldata)
        } else {
          message(paste("FAILED or EMPTY: Company layer:", lname, "from:", p))
          NULL
        }
      })
      app_data$company_layers_list <- Filter(Negate(is.null), company_layers_list_temp)
    } else {
      message("No .shp files found in the company directory.")
    }
  } else { message(paste("Company shapefile (Dissolved) directory NOT FOUND:", company_shapefiles_dir)) }
  message(paste("Loaded", length(app_data$company_layers_list), "valid company layers."))
  
  message(paste("Saving MERGED processed data (wells and layers) to:", processed_rds_file))
  tryCatch({
    saveRDS(app_data, file = processed_rds_file)
    message("Processed data (wells and layers) saved to RDS.")
  }, error = function(e){
    message(paste("Error saving RDS:", e$message))
  })
  # --- END OF SHAPEFILE LOADING LOGIC ---
  
}

wells_sf_global <- app_data$wells_sf
play_subplay_layers_list_global <- app_data$play_subplay_layers_list
company_layers_list_global <- app_data$company_layers_list

initial_play_subplay_layer_names <- if (length(play_subplay_layers_list_global) > 0) {
  sort(sapply(play_subplay_layers_list_global, function(x) if(!is.null(x$name)) x$name else NA_character_))
} else { character(0) }
initial_play_subplay_layer_names <- initial_play_subplay_layer_names[!is.na(initial_play_subplay_layer_names)]

initial_company_layer_names <- if (length(company_layers_list_global) > 0) {
  sort(sapply(company_layers_list_global, function(x) if(!is.null(x$name)) x$name else NA_character_))
} else { character(0) }
initial_company_layer_names <- initial_company_layer_names[!is.na(initial_company_layer_names)]

message("--- FINAL CHECK OF wells_sf_global BEFORE SERVER ---")
if (!is.null(wells_sf_global) && inherits(wells_sf_global, "sf")) {
  message(paste("  nrow(wells_sf_global):", nrow(wells_sf_global)))
  if ("ConfidentialType" %in% names(wells_sf_global)) {
    message(paste("  'ConfidentialType' column IS present in wells_sf_global. Sample values:", paste(head(unique(na.omit(wells_sf_global$ConfidentialType))), collapse=", ")))
  } else {
    message("  'ConfidentialType' column IS NOT present in wells_sf_global.")
  }
  if ("BH_Latitude" %in% names(wells_sf_global)) {
    message(paste("  'BH_Latitude' column IS present. Sample:", head(na.omit(wells_sf_global$BH_Latitude), 3)))
  } else {
    message("  'BH_Latitude' column IS NOT present.")
  }
  if ("BH_Longitude" %in% names(wells_sf_global)) {
    message(paste("  'BH_Longitude' column IS present. Sample:", head(na.omit(wells_sf_global$BH_Longitude), 3)))
  } else {
    message("  'BH_Longitude' column IS NOT present.")
  }
  if ("LateralLength" %in% names(wells_sf_global)) { # Check for LateralLength
    message(paste("  'LateralLength' column IS present. Sample:", head(na.omit(wells_sf_global$LateralLength), 3)))
  } else {
    message("  'LateralLength' column IS NOT present.")
  }
  if ("FirstProdDate" %in% names(wells_sf_global) && inherits(wells_sf_global$FirstProdDate, "Date")) {
    message(paste("  'FirstProdDate' column IS present and is a Date. Sample:", head(na.omit(wells_sf_global$FirstProdDate), 3)))
  } else {
    message("  'FirstProdDate' column IS NOT present or not a Date type.")
  }
} else { message("  wells_sf_global is NULL or not an sf object before server starts.") }
message(paste("  Initial Play/Subplay Layers for Picker:", length(initial_play_subplay_layer_names), "Names:", paste(initial_play_subplay_layer_names, collapse=", ")))
message(paste("  Initial Company Layers for Picker:", length(initial_company_layer_names), "Names:", paste(initial_company_layer_names, collapse=", ")))


# --- UI ---
ui <- fluidPage(
  tags$head(tags$style(HTML(".shiny-notification { position:fixed; top: calc(5%); left: calc(50% - 150px); width: 300px; z-index: 2000 !important; }"))),
  titlePanel("Interactive Well and Acreage Map Application (DB V24 - Well Sticks & Province Filter)"),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("Well Selection Criteria"),
      pickerInput("operator_filter", "Operator:", choices = c("Loading..."=""), selected = NULL, multiple = TRUE, options = pickerOptions(actionsBox = TRUE, liveSearch = TRUE, noneSelectedText = "Filter by Operator...", virtualScroll = TRUE, size=10)),
      pickerInput("formation_filter", "Formation:", choices = c("Loading..."=""), selected = NULL, multiple = TRUE, options = pickerOptions(actionsBox = TRUE, liveSearch = TRUE, noneSelectedText = "Filter by Formation...", virtualScroll = TRUE, size=10)),
      pickerInput("field_filter", "Field:", choices = c("Loading..."=""), selected = NULL, multiple = TRUE, options = pickerOptions(actionsBox = TRUE, liveSearch = TRUE, noneSelectedText = "Filter by Field...", virtualScroll = TRUE, size=10)),
      pickerInput("province_filter", "Province/State:", choices = c("Loading..."=""), selected = NULL, multiple = TRUE, options = pickerOptions(actionsBox = TRUE, liveSearch = TRUE, noneSelectedText = "Filter by Province/State...", virtualScroll = TRUE, size=5)),
      dateRangeInput("well_date_filter", "Filter by First Production Date:",
                     start = Sys.Date() - years(10), end = Sys.Date(),
                     min = as.Date("1900-01-01"), max = Sys.Date(),
                     format = "yyyy-mm-dd", startview = "year", width="100%"),
      actionButton("update_map", "Apply Filters & Update Map", class = "btn-primary btn-block"),
      actionButton("reset_filters", "Reset All Filters", class = "btn-block"),
      actionButton("reconnect_db_button", "Reconnect to Database", class = "btn-warning btn-block", style="margin-top: 10px;"), # Reconnect Button
      hr(),
      h5(strong(htmlOutput("well_count_display"))),
      hr(),
      h5("Map Layers:"),
      pickerInput("play_subplay_filter", "Play/Subplay Boundaries:", choices = initial_play_subplay_layer_names, selected = NULL, multiple = TRUE, options = pickerOptions(actionsBox = TRUE, liveSearch = TRUE, noneSelectedText = "None", virtualScroll = TRUE, size=5)),
      pickerInput("company_acreage_filter", "Company Acreage:", choices = initial_company_layer_names, selected = NULL, multiple = TRUE, options = pickerOptions(actionsBox = TRUE, liveSearch = TRUE, noneSelectedText = "None", virtualScroll = TRUE, size=5))
    ),
    mainPanel(
      width = 9,
      tabsetPanel(
        id = "main_tabs",
        tabPanel("Well Map", leafletOutput("well_map", height = "85vh")),
        tabPanel("Production Analysis",
                 fluidRow(
                   column(12,
                          pickerInput("product_type_filter_analysis", "Filter Analyses by Product Type:",
                                      choices = c("Oil" = "OIL", "Condensate" = "CND", "Gas" = "GAS", "BOE (Oil+Cnd+Gas)" = "BOE"),
                                      selected = c("OIL", "CND", "GAS", "BOE"), # Default to all including BOE
                                      multiple = TRUE,
                                      options = pickerOptions(actionsBox = TRUE, noneSelectedText = "Select Product(s)..."))
                   )
                 ),
                 hr(),
                 tabsetPanel(
                   id = "prod_analysis_tabs",
                   tabPanel("Single Well Analysis",
                            h4("Daily Production Rate"),
                            selectInput("selected_well_for_prod", "Select Well for Production:", choices = c("Apply filters and click a well on map or select here" = "")),
                            uiOutput("production_date_slider_ui"),
                            plotlyOutput("production_plot", height = "45vh"),
                            hr(),
                            h5("Production Data Table (for selected well & date range)"),
                            downloadButton("download_prod_data", "Download Table as CSV"),
                            DT::dataTableOutput("production_table")
                   ),
                   tabPanel("Filtered Group Cumulative",
                            h4("Time Normalized Production by Group (Map-Filtered Wells)"),
                            p("This analysis uses wells currently displayed on the map based on the main filters. Production is normalized by reported lateral length (if available). Assumes lateral length is in feet."),
                            fluidRow(
                              column(6,
                                     selectInput("filtered_group_breakout_by", "Normalize & Group By:",
                                                 choices = c("Operator" = "OperatorName",
                                                             "Formation" = "Formation",
                                                             "Field" = "FieldName",
                                                             "Province/State" = "ProvinceState",
                                                             "First Prod Year" = "FirstProdYear"),
                                                 selected = "OperatorName")
                              ),
                              column(6,
                                     actionButton("calculate_filtered_cumulative", "Calculate Normalized & Cumulative Rates", class = "btn-info btn-block", style="margin-top: 25px;") # Align button
                              )
                            ),
                            hr(),
                            uiOutput("filtered_group_plot_title_normalized"), # Dynamic title
                            plotOutput("filtered_group_cumulative_plot_normalized", height = "45vh"),
                            hr(),
                            uiOutput("filtered_group_plot_title_cumulative_boe"), # Dynamic title for cumulative
                            plotOutput("filtered_group_cumulative_plot_cumulative_boe", height = "45vh"),
                            hr(),
                            # --- NEW CHART UI ---
                            uiOutput("filtered_group_plot_title_calendar_rate"),
                            plotOutput("filtered_group_calendar_rate_plot", height = "45vh"),
                            hr(),
                            # --- END NEW CHART UI ---
                            h5("Filtered Group Production Data Summary (by Selected Group)"),
                            downloadButton("download_filtered_group_prod_data", "Download Summary as CSV"),
                            DT::dataTableOutput("filtered_group_production_table")
                   ),
                   tabPanel("Type Curve Analysis (Arps)",
                            h4("Arps Decline Curve (Peak Normalized - Daily Rates) for Map-Filtered Wells"),
                            p("Generates an Arps type curve based on aligning wells by their peak production month (using average daily rates). Uses wells currently displayed on the map and respects the 'Filter Analyses by Product Type' selection above."),
                            fluidRow(
                              column(6, selectInput("arps_product_type", "Select Product for Arps:", choices = c("Oil/Condensate" = "Oil", "Gas" = "Gas"))),
                              column(6, selectInput("arps_model_type", "Select Arps Model:", choices = c("Hyperbolic" = "hyperbolic", "Exponential" = "exponential", "Harmonic" = "harmonic")))
                            ),
                            actionButton("generate_type_curve", "Generate Type Curve", class = "btn-info btn-block"),
                            hr(),
                            plotOutput("arps_type_curve_plot", height="50vh"),
                            h5("Fitted Arps Parameters, EUR & Decline Summary:"),
                            verbatimTextOutput("arps_parameters_output"),
                            hr(),
                            h5("Aggregated Data Used for Type Curve (Avg Daily Rate vs. Months Since Peak)"),
                            DT::dataTableOutput("arps_data_table")
                   ),
                   tabPanel("Operator Group Cumulative",
                            h4("Operator Group Average Daily Production Rate"),
                            p(strong("Note:"), " This tab shows gross production for selected operators over a specific date range, independent of map filters. For operator comparisons within the main filter context (Formation, Field, Province, Date), please use the 'Filtered Group Cumulative' tab."),
                            pickerInput("group_operator_filter", "Select Operator(s) to Group:",
                                        choices = c("Loading..."=""), selected = NULL, multiple = TRUE,
                                        options = pickerOptions(actionsBox = TRUE, liveSearch = TRUE, noneSelectedText = "Select Operator(s)...", virtualScroll = TRUE, size=5, maxOptions = 5)),
                            dateRangeInput("group_prod_date_range", "Select Date Range for Grouped Analysis:",
                                           start = Sys.Date() - years(5), end = Sys.Date(),
                                           min = as.Date("1950-01-01"), max = Sys.Date(),
                                           format = "yyyy-mm-dd", startview = "year"),
                            actionButton("update_group_plot", "Update Operator Group Plot", class = "btn-info"),
                            hr(),
                            plotOutput("grouped_cumulative_plot", height = "50vh"),
                            h5("Operator Group Production Data Summary (Monthly, Cumulative, and Avg Daily Rates)"),
                            downloadButton("download_group_prod_data", "Download Operator Group Summary as CSV"),
                            DT::dataTableOutput("grouped_production_table")
                   )
                 )
        )
      )
    )
  )
)

# --- Server Logic ---
server <- function(input, output, session) {
  
  wells_sf <- wells_sf_global
  play_subplay_layers_list <- play_subplay_layers_list_global
  company_layers_list <- company_layers_list_global
  
  reactive_vals <- reactiveValues(
    wells_to_display = sf::st_sf(geometry = sf::st_sfc(), crs = 4326),
    has_map_been_updated_once = FALSE,
    current_selected_gsl_uwi_std = NULL,
    min_prod_date = as.Date("1900-01-01"),
    max_prod_date = Sys.Date(),
    min_first_prod_date_overall = as.Date("1900-01-01"),
    max_first_prod_date_overall = Sys.Date()
  )
  
  # Initial population of pickers (non-cascading)
  observe({
    req(wells_sf_global, nrow(wells_sf_global) > 0)
    message("SERVER OBSERVE (Initial Picker Population & Date Slider Range): Entered observer.")
    
    op_choices_init <- prepare_filter_choices(wells_sf_global$OperatorName, "OperatorName (initial)")
    form_choices_init <- prepare_filter_choices(wells_sf_global$Formation, "Formation (initial)")
    fld_choices_init <- prepare_filter_choices(wells_sf_global$FieldName, "FieldName (initial)")
    prov_choices_init <- prepare_filter_choices(wells_sf_global$ProvinceState, "ProvinceState (initial)")
    
    updatePickerInput(session, "operator_filter", choices = if(length(op_choices_init)>0) op_choices_init else c("No Operators Found" = ""), selected = NULL)
    updatePickerInput(session, "group_operator_filter", choices = if(length(op_choices_init)>0) op_choices_init else c("No Operators Found" = ""), selected = NULL)
    updatePickerInput(session, "formation_filter", choices = if(length(form_choices_init)>0) form_choices_init else c("No Formations Found" = ""), selected = NULL)
    updatePickerInput(session, "field_filter", choices = if(length(fld_choices_init)>0) fld_choices_init else c("No Fields Found" = ""), selected = NULL)
    updatePickerInput(session, "province_filter", choices = if(length(prov_choices_init)>0) prov_choices_init else c("No Provinces Found" = ""), selected = NULL)
    
    updatePickerInput(session, "play_subplay_filter",
                      choices = if(length(initial_play_subplay_layer_names)>0) initial_play_subplay_layer_names else c("No Layers Loaded" = ""),
                      selected = NULL)
    updatePickerInput(session, "company_acreage_filter",
                      choices = if(length(initial_company_layer_names)>0) initial_company_layer_names else c("No Layers Loaded" = ""),
                      selected = NULL)
    
    if (nrow(wells_sf_global) > 0 && "FirstProdDate" %in% names(wells_sf_global) && inherits(wells_sf_global$FirstProdDate, "Date") && sum(!is.na(wells_sf_global$FirstProdDate)) > 0) {
      min_fp_date <- min(wells_sf_global$FirstProdDate, na.rm = TRUE)
      max_fp_date <- max(wells_sf_global$FirstProdDate, na.rm = TRUE)
      reactive_vals$min_first_prod_date_overall <- min_fp_date
      reactive_vals$max_first_prod_date_overall <- max_fp_date
      default_start_date <- max(min_fp_date, max_fp_date - years(10), na.rm = TRUE)
      updateDateRangeInput(session, "well_date_filter",
                           min = min_fp_date, max = max_fp_date,
                           start = default_start_date, end = max_fp_date)
    } else {
      updateDateRangeInput(session, "well_date_filter",
                           min = as.Date("1900-01-01"), max = Sys.Date(),
                           start = Sys.Date() - years(10), end = Sys.Date())
    }
    message("SERVER: Initial Picker choices and date slider updated (non-cascading).")
  })
  
  observeEvent(input$reset_filters, {
    showNotification("Resetting all filters...", type = "message", duration=2, id="resetNotify")
    
    updatePickerInput(session, "operator_filter", selected = character(0))
    updatePickerInput(session, "formation_filter", selected = character(0))
    updatePickerInput(session, "field_filter", selected = character(0))
    updatePickerInput(session, "province_filter", selected = character(0))
    updatePickerInput(session, "company_acreage_filter", selected = character(0))
    updatePickerInput(session, "play_subplay_filter", selected = character(0))
    updatePickerInput(session, "group_operator_filter", selected = character(0))
    updatePickerInput(session, "product_type_filter_analysis", selected = c("OIL", "CND", "GAS", "BOE"))
    
    default_start_date_reset <- max(reactive_vals$min_first_prod_date_overall, reactive_vals$max_first_prod_date_overall - years(10), na.rm = TRUE)
    if (!is.finite(default_start_date_reset)) default_start_date_reset <- Sys.Date() - years(10)
    updateDateRangeInput(session, "well_date_filter",
                         start = default_start_date_reset,
                         end = reactive_vals$max_first_prod_date_overall)
    
    reactive_vals$wells_to_display <- sf::st_sf(geometry=sf::st_sfc(), crs=4326)
    reactive_vals$has_map_been_updated_once <- FALSE
    reactive_vals$current_selected_gsl_uwi_std <- NULL
    updateSelectInput(session, "selected_well_for_prod",
                      choices = c("Apply filters and click a well on map or select here" = ""),
                      selected = "")
    showNotification("Filters reset. Apply filters to display wells.", type = "warning", duration=3, id="resetCompleteNotify")
  })
  
  # Database Reconnect Logic
  observeEvent(input$reconnect_db_button, {
    showNotification("Attempting to reconnect to database...", type="message", id="dbReconnectMsg", duration = NULL)
    if (!is.null(con) && dbIsValid(con)) {
      tryCatch({
        dbDisconnect(con)
        message("Disconnected by user for reconnect.")
      }, error = function(e) {
        message(paste("Error during user-initiated disconnect:", e$message))
      })
    }
    con <<- connect_to_db() # Use <<- to assign to global con
    if (!is.null(con) && dbIsValid(con)) {
      removeNotification("dbReconnectMsg")
      showNotification("Successfully reconnected to the database.", type="message", duration = 3)
    } else {
      removeNotification("dbReconnectMsg")
      showNotification("Failed to reconnect to the database. Please check console for details.", type="error", duration = 5)
    }
  })
  
  observeEvent(input$update_map, {
    req(wells_sf)
    showNotification("Applying filters and updating map...", type = "message", id="mapUpdateMsg", duration = NULL)
    if (is.null(wells_sf) || !inherits(wells_sf, "sf")) {
      reactive_vals$wells_to_display <- sf::st_sf(geometry = sf::st_sfc(), crs = 4326)
      reactive_vals$has_map_been_updated_once <- TRUE
      removeNotification("mapUpdateMsg")
      showNotification("Well data source is not available.", type="error", duration=3)
      return()
    }
    df <- wells_sf
    if (nrow(df) == 0) {
      reactive_vals$wells_to_display <- sf::st_sf(geometry = sf::st_sfc(), crs = 4326)
      reactive_vals$has_map_been_updated_once <- TRUE
      removeNotification("mapUpdateMsg")
      showNotification("No well data available to filter.", type="warning", duration=3)
      return()
    }
    current_operator_filter <- input$operator_filter[!input$operator_filter %in% c("", "Loading...", "No Operators Found", "No Well Data")]
    current_formation_filter <- input$formation_filter[!input$formation_filter %in% c("", "Loading...", "No Formations Found", "No Well Data")]
    current_field_filter <- input$field_filter[!input$field_filter %in% c("", "Loading...", "No Fields Found", "No Well Data")]
    current_province_filter <- input$province_filter[!input$province_filter %in% c("", "Loading...", "No Provinces Found", "No Well Data")]
    
    current_date_filter_start <- input$well_date_filter[1]
    current_date_filter_end <- input$well_date_filter[2]
    
    if (length(current_operator_filter) > 0) {
      if ("OperatorName" %in% names(df)) df <- df %>% filter(OperatorName %in% current_operator_filter)
    }
    if (length(current_formation_filter) > 0) {
      if ("Formation" %in% names(df)) df <- df %>% filter(Formation %in% current_formation_filter)
    }
    if (length(current_field_filter) > 0) {
      if ("FieldName" %in% names(df)) df <- df %>% filter(FieldName %in% current_field_filter)
    }
    if (length(current_province_filter) > 0) {
      if ("ProvinceState" %in% names(df)) df <- df %>% filter(ProvinceState %in% current_province_filter)
    }
    if ("FirstProdDate" %in% names(df) && inherits(df$FirstProdDate, "Date")) {
      if (!is.na(current_date_filter_start) && !is.na(current_date_filter_end)) {
        df <- df %>% filter(FirstProdDate >= current_date_filter_start & FirstProdDate <= current_date_filter_end)
      } else {
        message("Date filter not applied as start or end date is NA.")
      }
    } else {
      message("FirstProdDate column not found or not Date type, skipping date filter.")
    }
    reactive_vals$wells_to_display <- df
    reactive_vals$has_map_been_updated_once <- TRUE
    
    well_choices_for_prod <- c("Apply filters or click a well" = "")
    if (nrow(df) > 0 && all(c("GSL_UWI_Std", "WellName", "UWI") %in% names(df))) {
      clean_for_display <- function(text_vector) {
        if (is.null(text_vector)) return(rep("[Missing Data]", length(text_vector)))
        text_vector <- as.character(text_vector)
        cleaned_text <- iconv(text_vector, from = "", to = "UTF-8", sub = "?")
        cleaned_text[is.na(cleaned_text) & !is.na(text_vector)] <- "[Encoding Issue]"
        cleaned_text[is.na(cleaned_text)] <- "[Missing]"
        return(cleaned_text)
      }
      well_name_cleaned <- clean_for_display(df$WellName)
      uwi_cleaned <- clean_for_display(df$UWI)
      gsl_uwi_std_values <- df$GSL_UWI_Std
      valid_gsl_uwis <- !is.na(gsl_uwi_std_values) & gsl_uwi_std_values != ""
      if(any(valid_gsl_uwis)) {
        display_names_filtered <- paste(
          str_trunc(well_name_cleaned[valid_gsl_uwis], width = 30, side = "right", ellipsis = "..."),
          "- UWI:",
          str_trunc(uwi_cleaned[valid_gsl_uwis], width = 15, side = "right", ellipsis = "...")
        )
        well_choices_for_prod <- setNames(gsl_uwi_std_values[valid_gsl_uwis], display_names_filtered)
        if(anyDuplicated(names(well_choices_for_prod))) {
          names(well_choices_for_prod) <- make.unique(names(well_choices_for_prod))
        }
        well_choices_for_prod <- c("Select a well from filtered list..." = "", well_choices_for_prod)
      } else { well_choices_for_prod <- c("No wells with valid IDs in filter" = "") }
    } else { well_choices_for_prod <- c("Filtered list empty or key IDs missing" = "") }
    current_selection <- isolate(reactive_vals$current_selected_gsl_uwi_std)
    if(!is.null(current_selection) && current_selection %in% unname(well_choices_for_prod[well_choices_for_prod != ""])){
      updateSelectInput(session, "selected_well_for_prod", choices = well_choices_for_prod, selected = current_selection)
    } else {
      updateSelectInput(session, "selected_well_for_prod", choices = well_choices_for_prod, selected = "")
      if(!is.null(current_selection) && current_selection != "") reactive_vals$current_selected_gsl_uwi_std <- NULL
    }
    removeNotification("mapUpdateMsg")
    showNotification(paste("Map updated with", format(nrow(df), big.mark=","), "wells."), type="message", duration=3, id="mapUpdateSuccessNotify")
  })
  
  output$well_count_display <- renderUI({
    if (!reactive_vals$has_map_been_updated_once && nrow(reactive_vals$wells_to_display) == 0) {
      return(HTML("Apply filters to see well counts."))
    }
    displayed_wells <- reactive_vals$wells_to_display
    total_count <- nrow(displayed_wells)
    
    confidential_count <- 0
    if ("ConfidentialType" %in% names(displayed_wells) && total_count > 0) {
      confidential_values <- displayed_wells$ConfidentialType
      confidential_count <- sum(toupper(confidential_values) == "CONFIDENTIAL", na.rm = TRUE)
    }
    
    HTML(paste0("Total Wells Displayed: ", format(total_count, big.mark = ","), "<br/>",
                "Confidential Wells: ", format(confidential_count, big.mark = ",")))
  })
  
  output$well_map <- renderLeaflet({
    leaflet() %>%
      addProviderTiles(providers$CartoDB.Positron, group = "Simple Map") %>%
      addProviderTiles(providers$OpenStreetMap.Mapnik, group = "OpenStreetMap") %>%
      addProviderTiles(providers$Esri.WorldImagery, group = "Satellite") %>%
      setView(lng = -106, lat = 55, zoom = 4) %>%
      addLayersControl(
        baseGroups = c("Simple Map", "OpenStreetMap", "Satellite"),
        overlayGroups = c("Wells", "Well Sticks", "Play/Subplay Acreage", "Company Acreage"),
        options = layersControlOptions(collapsed = FALSE)
      ) %>%
      addScaleBar(position = "bottomleft") %>%
      leaflet.extras::addFullscreenControl()
  })
  
  observe({
    df_map <- reactive_vals$wells_to_display
    selected_co_acreage_names <- input$company_acreage_filter
    selected_ps_acreage_names <- input$play_subplay_filter
    
    proxy <- leafletProxy("well_map", data = df_map) %>%
      clearMarkers() %>%
      clearMarkerClusters() %>%
      clearShapes() %>%
      clearControls() %>%
      addLayersControl(
        baseGroups = c("Simple Map", "OpenStreetMap", "Satellite"),
        overlayGroups = c("Wells", "Well Sticks", "Play/Subplay Acreage", "Company Acreage"),
        options = layersControlOptions(collapsed = FALSE)
      ) %>%
      addScaleBar(position = "bottomleft") %>%
      leaflet.extras::addFullscreenControl(position = "topleft")
    
    current_overlay_groups <- c("Wells", "Well Sticks")
    acreage_legend_labels <- character(0)
    acreage_legend_colors <- character(0)
    
    # Acreage Layers (Polygons)
    if (!is.null(selected_ps_acreage_names) && length(selected_ps_acreage_names) > 0) {
      current_overlay_groups <- c(current_overlay_groups, "Play/Subplay Acreage")
      selected_layers_data_ps <- Filter(function(layer) !is.null(layer$name) && layer$name %in% selected_ps_acreage_names, play_subplay_layers_list)
      if (length(selected_layers_data_ps) > 0) {
        ps_colors_map <- setNames(RColorBrewer::brewer.pal(n = max(3, min(length(selected_layers_data_ps), 9)), name = "Set2"),
                                  sapply(selected_layers_data_ps, function(x) x$name))
        
        for (i in seq_along(selected_layers_data_ps)) {
          layer_info <- selected_layers_data_ps[[i]]
          layer_name_escaped <- htmltools::htmlEscape(layer_info$name)
          layer_color <- ps_colors_map[[layer_info$name]]
          
          if (!is.null(layer_info$data) && inherits(layer_info$data, "sf") && nrow(layer_info$data) > 0) {
            layer_attributes_ps <- sf::st_drop_geometry(layer_info$data)
            popups_for_layer_ps <- paste0("<b>Play/Subplay:</b> ", layer_name_escaped)
            if (nrow(layer_attributes_ps) > 0 && ncol(layer_attributes_ps) > 0) {
              attr_to_display_name_ps <- names(layer_attributes_ps)[1]
              if ("SUBPLAY" %in% toupper(names(layer_attributes_ps))) {
                attr_to_display_name_ps <- names(layer_attributes_ps)[toupper(names(layer_attributes_ps)) == "SUBPLAY"][1]
              } else if ("PLAY" %in% toupper(names(layer_attributes_ps))) {
                attr_to_display_name_ps <- names(layer_attributes_ps)[toupper(names(layer_attributes_ps)) == "PLAY"][1]
              } else if ("NAME" %in% toupper(names(layer_attributes_ps))) {
                attr_to_display_name_ps <- names(layer_attributes_ps)[toupper(names(layer_attributes_ps)) == "NAME"][1]
              }
              popups_for_layer_ps <- paste0("<b>Play/Subplay:</b> ", layer_name_escaped, "<br>",
                                            "<b>", htmltools::htmlEscape(attr_to_display_name_ps), ":</b> ", htmltools::htmlEscape(layer_attributes_ps[[attr_to_display_name_ps]]))
            }
            proxy %>% addPolygons(data = layer_info$data, group = "Play/Subplay Acreage",
                                  layerId = paste0("ps_",make.names(layer_info$name), "_feature_", seq_len(nrow(layer_info$data))),
                                  color = layer_color, weight = 1.5, fillColor = layer_color, fillOpacity = 0.15,
                                  popup = lapply(popups_for_layer_ps, htmltools::HTML),
                                  highlightOptions = highlightOptions(weight = 3, color = "white", bringToFront = TRUE, fillOpacity=0.3))
            acreage_legend_labels <- c(acreage_legend_labels, layer_name_escaped)
            acreage_legend_colors <- c(acreage_legend_colors, layer_color)
          }
        }
      }
    }
    if (!is.null(selected_co_acreage_names) && length(selected_co_acreage_names) > 0) {
      current_overlay_groups <- c(current_overlay_groups, "Company Acreage")
      selected_layers_data_co <- Filter(function(layer) !is.null(layer$name) && layer$name %in% selected_co_acreage_names, company_layers_list)
      if (length(selected_layers_data_co) > 0) {
        co_colors_map <- setNames(RColorBrewer::brewer.pal(n = max(3, min(length(selected_layers_data_co), 8)), name = "Pastel2"),
                                  sapply(selected_layers_data_co, function(x) x$name))
        for (i in seq_along(selected_layers_data_co)) {
          layer_info <- selected_layers_data_co[[i]]
          layer_name_escaped <- htmltools::htmlEscape(layer_info$name)
          layer_color <- co_colors_map[[layer_info$name]]
          
          if (!is.null(layer_info$data) && inherits(layer_info$data, "sf") && nrow(layer_info$data) > 0) {
            layer_attributes_co <- sf::st_drop_geometry(layer_info$data)
            popups_for_layer_co <- paste0("<b>Company Acreage:</b> ", layer_name_escaped)
            if (nrow(layer_attributes_co) > 0 && ncol(layer_attributes_co) > 0) {
              attr_to_display_name_co <- names(layer_attributes_co)[1]
              if ("COMPANY" %in% toupper(names(layer_attributes_co))) {
                attr_to_display_name_co <- names(layer_attributes_co)[toupper(names(layer_attributes_co)) == "COMPANY"][1]
              } else if ("OPERATOR" %in% toupper(names(layer_attributes_co))) {
                attr_to_display_name_co <- names(layer_attributes_co)[toupper(names(layer_attributes_co)) == "OPERATOR"][1]
              } else if ("NAME" %in% toupper(names(layer_attributes_co))) {
                attr_to_display_name_co <- names(layer_attributes_co)[toupper(names(layer_attributes_co)) == "NAME"][1]
              }
              popups_for_layer_co <- paste0("<b>Company Acreage:</b> ", layer_name_escaped, "<br>",
                                            "<b>", htmltools::htmlEscape(attr_to_display_name_co), ":</b> ", htmltools::htmlEscape(layer_attributes_co[[attr_to_display_name_co]]))
            }
            proxy %>% addPolygons(data = layer_info$data, group = "Company Acreage",
                                  layerId = paste0("co_",make.names(layer_info$name), "_feature_", seq_len(nrow(layer_info$data))),
                                  color = "black", weight = 1, fillColor = layer_color, fillOpacity = 0.35,
                                  popup = lapply(popups_for_layer_co, htmltools::HTML),
                                  highlightOptions = highlightOptions(weight = 3, color = "white", bringToFront = TRUE, fillOpacity=0.5))
            acreage_legend_labels <- c(acreage_legend_labels, layer_name_escaped)
            acreage_legend_colors <- c(acreage_legend_colors, layer_color)
          }
        }
      }
    }
    
    # Well Markers (Surface Points) and Well Sticks (Polylines)
    if (!is.null(df_map) && inherits(df_map, "sf") && nrow(df_map) > 0) {
      
      pal_op <- colorFactor("viridis", domain = NULL)
      if ("OperatorName" %in% names(df_map) && sum(!is.na(df_map$OperatorName)) > 0) {
        unique_operators <- unique(na.omit(df_map$OperatorName))
        if (length(unique_operators) > 0) {
          if (length(unique_operators) <= 12) {
            pal_op <- colorFactor(palette = RColorBrewer::brewer.pal(max(3, length(unique_operators)), "Paired"), domain = unique_operators)
          } else {
            pal_op <- colorFactor(palette = viridis::viridis(length(unique_operators)), domain = unique_operators)
          }
        }
      }
      
      well_layer_id_col_name <- if (!"GSL_UWI_Std" %in% names(df_map) || !is.character(df_map$GSL_UWI_Std)) {
        df_map$GSL_UWI_Std_for_map <- paste0("wellmarker_", seq_len(nrow(df_map)))
        "GSL_UWI_Std_for_map"
      } else { "GSL_UWI_Std" }
      
      base_popup <- paste0(
        "<b>UWI:</b> ", htmltools::htmlEscape(df_map$UWI), "<br>",
        "<b>Well Name:</b> ", htmltools::htmlEscape(df_map$WellName), "<br>",
        "<b>Operator:</b> ", htmltools::htmlEscape(df_map$OperatorName), "<br>",
        "<b>Formation:</b> ", htmltools::htmlEscape(df_map$Formation), "<br>",
        "<b>Field:</b> ", htmltools::htmlEscape(df_map$FieldName), "<br>",
        "<b>Status:</b> ", htmltools::htmlEscape(df_map$CurrentStatus), "<br>",
        "<b>First Prod Date:</b> ", htmltools::htmlEscape(as.character(df_map$FirstProdDate))
      )
      
      confidential_text_vec <- if ("ConfidentialType" %in% names(df_map)) {
        ifelse(!is.na(df_map$ConfidentialType),
               paste0("<br><b>Confidential:</b> ", htmltools::htmlEscape(df_map$ConfidentialType)),
               "")
      } else { rep("", nrow(df_map)) }
      
      bh_lat_text_vec <- if ("BH_Latitude" %in% names(df_map)) {
        ifelse(!is.na(df_map$BH_Latitude),
               paste0("<br><b>BH Lat:</b> ", round(df_map$BH_Latitude, 5)),
               "")
      } else { rep("", nrow(df_map)) }
      
      bh_lon_text_vec <- if ("BH_Longitude" %in% names(df_map)) {
        ifelse(!is.na(df_map$BH_Longitude),
               paste0("<br><b>BH Lon:</b> ", round(df_map$BH_Longitude, 5)),
               "")
      } else { rep("", nrow(df_map)) }
      
      well_popup_content <- paste0(base_popup, confidential_text_vec, bh_lat_text_vec, bh_lon_text_vec)
      
      proxy %>% addCircleMarkers(
        lng = df_map$SurfaceLongitude,
        lat = df_map$SurfaceLatitude,
        radius = 4,
        color = if("OperatorName" %in% names(df_map) && sum(!is.na(df_map$OperatorName)) > 0) pal_op(df_map$OperatorName) else "blue",
        stroke = TRUE, weight=1,
        fillOpacity = 0.6,
        popup = lapply(well_popup_content, htmltools::HTML),
        layerId = df_map[[well_layer_id_col_name]],
        group = "Wells",
        clusterOptions = markerClusterOptions(spiderfyOnMaxZoom = TRUE, showCoverageOnHover = TRUE, zoomToBoundsOnClick = TRUE)
      )
      
      wells_with_bh <- df_map[!is.na(df_map$BH_Latitude) & !is.na(df_map$BH_Longitude) &
                                !is.na(df_map$SurfaceLatitude) & !is.na(df_map$SurfaceLongitude), ]
      
      if (nrow(wells_with_bh) > 0) {
        for (i in 1:nrow(wells_with_bh)) {
          well_stick_data <- wells_with_bh[i, ]
          proxy %>% addPolylines(
            lng = c(well_stick_data$SurfaceLongitude, well_stick_data$BH_Longitude),
            lat = c(well_stick_data$SurfaceLatitude, well_stick_data$BH_Latitude),
            layerId = paste0(well_stick_data[[well_layer_id_col_name]], "_stick"),
            color = if("OperatorName" %in% names(well_stick_data) && sum(!is.na(well_stick_data$OperatorName)) > 0) pal_op(well_stick_data$OperatorName) else "red",
            weight = 2,
            opacity = 0.7,
            group = "Well Sticks"
          )
        }
      }
      
      if ("OperatorName" %in% names(df_map) && sum(!is.na(df_map$OperatorName)) > 0 && length(unique(stats::na.omit(df_map$OperatorName))) > 0 && length(unique(stats::na.omit(df_map$OperatorName))) <= 20) {
        proxy %>% addLegend("bottomright", pal = pal_op, values = ~OperatorName, title = "Operator", opacity = 1, layerId = "op_legend")
      }
    }
    
    # Add Acreage Legend if any acreage layers are selected
    if (length(acreage_legend_labels) > 0) {
      proxy %>% addLegend(
        position = "bottomleft", # Or another position
        colors = acreage_legend_colors,
        labels = acreage_legend_labels,
        title = "Acreage Layers",
        opacity = 0.7,
        layerId = "acreage_legend"
      )
    }
    
    # Update Layers Control with all potentially visible groups
    # This was already being done correctly.
  })
  
  observeEvent(input$well_map_marker_click, {
    event <- input$well_map_marker_click
    if (is.null(event$id)) return()
    
    cleaned_event_id <- sub("_stick$", "", event$id)
    
    if ("GSL_UWI_Std" %in% names(reactive_vals$wells_to_display) &&
        cleaned_event_id %in% reactive_vals$wells_to_display$GSL_UWI_Std) {
      reactive_vals$current_selected_gsl_uwi_std <- cleaned_event_id
      updateSelectInput(session, "selected_well_for_prod", selected = cleaned_event_id)
      showNotification(paste("Selected well ID:", cleaned_event_id, "for production analysis."), type="message", duration=4, id="wellSelectNotify")
    } else {
      message(paste("Map click ID not directly matched to a GSL_UWI_Std:", event$id))
    }
  })
  
  observeEvent(input$selected_well_for_prod, {
    selected_dropdown_uwi <- input$selected_well_for_prod
    if (!is.null(selected_dropdown_uwi) && selected_dropdown_uwi != "" &&
        (is.null(reactive_vals$current_selected_gsl_uwi_std) || selected_dropdown_uwi != reactive_vals$current_selected_gsl_uwi_std) ) {
      reactive_vals$current_selected_gsl_uwi_std <- selected_dropdown_uwi
    } else if (is.null(selected_dropdown_uwi) || selected_dropdown_uwi == "") {
      if (!is.null(reactive_vals$current_selected_gsl_uwi_std)) {
        reactive_vals$current_selected_gsl_uwi_std <- NULL
      }
    }
  }, ignoreNULL = FALSE, ignoreInit = TRUE)
  
  # --- Production Data Fetching and Plotting (Single well, Groups) ---
  fetched_production_data <- reactive({
    req(reactive_vals$current_selected_gsl_uwi_std, input$product_type_filter_analysis, cancelOutput = TRUE)
    selected_uwi_std_for_prod <- reactive_vals$current_selected_gsl_uwi_std
    selected_analysis_products <- input$product_type_filter_analysis
    
    if (is.null(selected_uwi_std_for_prod) || selected_uwi_std_for_prod == "") return(data.table())
    if (is.null(selected_analysis_products) || length(selected_analysis_products) == 0) {
      showNotification("Please select at least one product type for analysis.", type = "warning", duration=5)
      return(data.table())
    }
    
    message(paste0("--- fetched_production_data: Fetching for GSL_UWI_Std: '", selected_uwi_std_for_prod, "' ---"))
    if (is.null(con) || !dbIsValid(con)) {
      message("Attempting to (re)connect for on-demand production...")
      con <<- connect_to_db()
      if (is.null(con) || !dbIsValid(con)) { message("ERROR: DB connection failed for on-demand production."); return(data.table()) }
    }
    sql_single_well_prod <- glue::glue_sql(
      "SELECT GSL_UWI, YEAR, PRODUCT_TYPE, ACTIVITY_TYPE, ",
      "JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME, ",
      "MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, ",
      "SEP_VOLUME, OCT_VOLUME, NOV_VOLUME, DEC_VOLUME ",
      "FROM PDEN_VOL_BY_MONTH WHERE GSL_UWI = {selected_uwi_std_for_prod} ",
      "AND ACTIVITY_TYPE = 'PRODUCTION' AND PRODUCT_TYPE IN ('OIL', 'CND', 'GAS') AND ROWNUM <= 240",
      .con = con
    )
    well_prod_raw <- tryCatch({ data.table::as.data.table(dbGetQuery(con, sql_single_well_prod)) },
                              error = function(e) { warning(paste("Error fetching production for GSL_UWI_Std", selected_uwi_std_for_prod, ":", e$message)); return(data.table()) })
    message(paste("Rows returned from DB for GSL_UWI_Std", selected_uwi_std_for_prod, ":", nrow(well_prod_raw)))
    if (nrow(well_prod_raw) == 0) return(data.table())
    
    if("GSL_UWI" %in% names(well_prod_raw)) { well_prod_raw[, GSL_UWI_Std_from_query := standardize_uwi(GSL_UWI)] } else { well_prod_raw[, GSL_UWI_Std_from_query := selected_uwi_std_for_prod] }
    prod_dt_cleaned <- clean_df_colnames(well_prod_raw, paste("PDEN Volumes for", selected_uwi_std_for_prod))
    if ("GSL_UWI_STD_FROM_QUERY" %in% names(prod_dt_cleaned)) { setnames(prod_dt_cleaned, "GSL_UWI_STD_FROM_QUERY", "GSL_UWI_Std")
    } else if ("GSL_UWI" %in% names(prod_dt_cleaned) && !"GSL_UWI_Std" %in% names(prod_dt_cleaned)) { prod_dt_cleaned[, GSL_UWI_Std := standardize_uwi(GSL_UWI)]; prod_dt_cleaned[, GSL_UWI := NULL]
    } else if (!"GSL_UWI_Std" %in% names(prod_dt_cleaned)) { prod_dt_cleaned[, GSL_UWI_Std := selected_uwi_std_for_prod] }
    year_col_actual <- "YEAR"; product_type_col_actual <- "PRODUCT_TYPE"
    
    oil_prod_val <- "OIL"
    cnd_prod_val <- "CND"
    gas_prod_val <- "GAS"
    
    month_abbrs_upper <- toupper(month.abb); monthly_vol_cols <- character(0)
    prod_dt_colnames_upper <- names(prod_dt_cleaned)
    for(m_abbr in month_abbrs_upper){ potential_col_upper <- paste0(m_abbr, "_VOLUME"); if(potential_col_upper %in% prod_dt_colnames_upper) { monthly_vol_cols <- c(monthly_vol_cols, potential_col_upper) } }
    required_id_cols <- c("GSL_UWI_Std", year_col_actual, product_type_col_actual)
    if(!(length(monthly_vol_cols) == 12 && all(required_id_cols %in% names(prod_dt_cleaned)))) { warning("Not all required columns found for melting production."); return(data.table()) }
    for(col_name in monthly_vol_cols) { if (!is.numeric(prod_dt_cleaned[[col_name]])) prod_dt_cleaned[, (col_name) := as.numeric(get(col_name))] }
    if (!is.numeric(prod_dt_cleaned[[year_col_actual]])) prod_dt_cleaned[, (year_col_actual) := as.numeric(get(year_col_actual))]
    if (!is.character(prod_dt_cleaned[[product_type_col_actual]])) prod_dt_cleaned[, (product_type_col_actual) := as.character(get(product_type_col_actual))]
    
    prod_long <- data.table::melt(prod_dt_cleaned, id.vars = required_id_cols, measure.vars = monthly_vol_cols, variable.name = "MONTH_VOLUME_COL", value.name = "Volume", na.rm = FALSE, verbose = FALSE)
    
    # Convert units first
    prod_long[toupper(get(product_type_col_actual)) %in% c(toupper(oil_prod_val), toupper(cnd_prod_val)), Volume_Converted := Volume * M3_TO_BBL]
    prod_long[toupper(get(product_type_col_actual)) == toupper(gas_prod_val), Volume_Converted := Volume * E3M3_TO_MCF]
    prod_long[is.na(Volume_Converted), Volume_Converted := 0]
    
    prod_long <- prod_long[!is.na(Volume_Converted) & Volume_Converted != 0]
    if(nrow(prod_long) == 0) return(data.table())
    
    prod_long[, Month_Num := match(toupper(substr(MONTH_VOLUME_COL, 1, 3)), month_abbrs_upper)]
    prod_long <- prod_long[!is.na(get(year_col_actual)) & !is.na(Month_Num)]
    prod_long[, PROD_DATE := tryCatch(as.Date(paste(get(year_col_actual), Month_Num, 1, sep="-"), format="%Y-%m-%d"), error = function(e) as.Date(NA))]
    prod_long <- prod_long[!is.na(PROD_DATE)]
    
    prod_long[, MonthlyOilTrueBBL_raw := fifelse(toupper(get(product_type_col_actual)) == toupper(oil_prod_val), Volume_Converted, 0)]
    prod_long[, MonthlyCondensateBBL_raw := fifelse(toupper(get(product_type_col_actual)) == toupper(cnd_prod_val), Volume_Converted, 0)]
    prod_long[, MonthlyGasMCF_raw := fifelse(toupper(get(product_type_col_actual)) == toupper(gas_prod_val), Volume_Converted, 0)]
    
    use_oil_for_boe <- "OIL" %in% selected_analysis_products || "BOE" %in% selected_analysis_products
    use_cnd_for_boe <- "CND" %in% selected_analysis_products || "BOE" %in% selected_analysis_products
    use_gas_for_boe <- "GAS" %in% selected_analysis_products || "BOE" %in% selected_analysis_products
    
    aggregated_prod_for_well <- prod_long[, .(
      MonthlyOilTrueBBL = if("OIL" %in% selected_analysis_products) sum(MonthlyOilTrueBBL_raw, na.rm = TRUE) else 0,
      MonthlyCondensateBBL = if("CND" %in% selected_analysis_products) sum(MonthlyCondensateBBL_raw, na.rm = TRUE) else 0,
      MonthlyGasMCF = if("GAS" %in% selected_analysis_products) sum(MonthlyGasMCF_raw, na.rm = TRUE) else 0,
      MonthlyOilForBOE = if(use_oil_for_boe) sum(MonthlyOilTrueBBL_raw, na.rm = TRUE) else 0,
      MonthlyCondensateForBOE = if(use_cnd_for_boe) sum(MonthlyCondensateBBL_raw, na.rm = TRUE) else 0,
      MonthlyGasForBOE = if(use_gas_for_boe) sum(MonthlyGasMCF_raw, na.rm = TRUE) else 0
    ), by = .(GSL_UWI_Std, PROD_DATE)][order(PROD_DATE)]
    
    return(aggregated_prod_for_well)
  })
  
  production_date_range <- reactive({
    prod_data <- fetched_production_data()
    if (nrow(prod_data) > 0 && "PROD_DATE" %in% names(prod_data)) {
      min_d <- min(prod_data$PROD_DATE, na.rm = TRUE)
      max_d <- max(prod_data$PROD_DATE, na.rm = TRUE)
      if(is.finite(min_d) && is.finite(max_d) && min_d <= max_d) { return(c(min_d, max_d)) }
    }
    return(c(as.Date("1900-01-01"), Sys.Date()))
  })
  output$production_date_slider_ui <- renderUI({
    date_rng <- production_date_range()
    sliderInput("production_date_filter", "Filter Production Dates:", min = date_rng[1], max = date_rng[2], value = date_rng, timeFormat = "%b %Y", width = "100%")
  })
  
  processed_production_for_plotting <- reactive({
    prod_data <- fetched_production_data()
    req(input$production_date_filter)
    if (nrow(prod_data) == 0) return(data.table())
    prod_data_filtered <- prod_data[PROD_DATE >= input$production_date_filter[1] & PROD_DATE <= input$production_date_filter[2]]
    if (nrow(prod_data_filtered) == 0) return(data.table())
    prod_data_filtered <- prod_data_filtered[order(PROD_DATE)]
    
    prod_data_filtered[, CumOilTrueBBL := cumsum(MonthlyOilTrueBBL), by = .(GSL_UWI_Std)]
    prod_data_filtered[, CumCondensateBBL := cumsum(MonthlyCondensateBBL), by = .(GSL_UWI_Std)]
    prod_data_filtered[, CumGasMCF := cumsum(MonthlyGasMCF), by = .(GSL_UWI_Std)]
    
    prod_data_filtered[, DaysInMonth := lubridate::days_in_month(PROD_DATE)]
    prod_data_filtered[, OilTrueRateBBLD := MonthlyOilTrueBBL / DaysInMonth]
    prod_data_filtered[, CondensateRateBBLD := MonthlyCondensateBBL / DaysInMonth]
    prod_data_filtered[, GasRateMCFD := MonthlyGasMCF / DaysInMonth]
    
    prod_data_filtered[, BOERateBBLD := (MonthlyOilForBOE + MonthlyCondensateForBOE + (MonthlyGasForBOE / MCF_PER_BOE)) / DaysInMonth]
    return(prod_data_filtered)
  })
  
  output$production_plot <- renderPlotly({
    plot_data <- processed_production_for_plotting()
    if (is.null(plot_data) || nrow(plot_data) == 0 || !"PROD_DATE" %in% names(plot_data)) {
      p <- ggplot() + labs(title = "No well selected or no production data available for selected product(s).", x=NULL, y=NULL) + theme_void()
      return(ggplotly(p))
    }
    plot_title_detail <- ""
    if (!is.null(reactive_vals$current_selected_gsl_uwi_std) && reactive_vals$current_selected_gsl_uwi_std != "" &&
        !is.null(wells_sf) && nrow(wells_sf) > 0) {
      well_details_list <- wells_sf[wells_sf$GSL_UWI_Std == reactive_vals$current_selected_gsl_uwi_std, ]
      if(nrow(well_details_list) > 0) {
        plot_title_detail <- paste0(well_details_list$WellName[1], " (UWI: ", well_details_list$UWI[1], ")")
      } else { plot_title_detail <- paste0("GSL_UWI: ", reactive_vals$current_selected_gsl_uwi_std) }
    }
    plot_main_title <- paste("Daily Production Rate:", plot_title_detail)
    
    measure_vars_plot <- character()
    if ("OIL" %in% input$product_type_filter_analysis && any(plot_data$OilTrueRateBBLD > 0, na.rm = TRUE)) measure_vars_plot <- c(measure_vars_plot, "OilTrueRateBBLD")
    if ("CND" %in% input$product_type_filter_analysis && any(plot_data$CondensateRateBBLD > 0, na.rm = TRUE)) measure_vars_plot <- c(measure_vars_plot, "CondensateRateBBLD")
    if ("GAS" %in% input$product_type_filter_analysis && any(plot_data$GasRateMCFD > 0, na.rm = TRUE)) measure_vars_plot <- c(measure_vars_plot, "GasRateMCFD")
    if ("BOE" %in% input$product_type_filter_analysis && any(plot_data$BOERateBBLD > 0, na.rm = TRUE)) {
      measure_vars_plot <- c(measure_vars_plot, "BOERateBBLD")
    }
    measure_vars_plot <- unique(measure_vars_plot)
    
    if(length(measure_vars_plot) == 0){
      p <- ggplot() + labs(title = paste(plot_main_title, "\n(No production rates to display for selected product(s))"), x="Date", y="Rate") + theme_minimal()
      return(ggplotly(p))
    }
    
    plot_data_long <- data.table::melt(plot_data, id.vars = c("GSL_UWI_Std", "PROD_DATE"),
                                       measure.vars = measure_vars_plot,
                                       variable.name = "ProductRateType", value.name = "Rate")
    plot_data_long_filtered <- plot_data_long[Rate > 0 & is.finite(Rate)]
    
    if(nrow(plot_data_long_filtered) == 0){
      p <- ggplot() + labs(title = paste(plot_main_title, "\n(No positive production rates to display for selected product(s))"), x="Date", y="Rate") + theme_minimal()
      return(ggplotly(p))
    }
    
    plot_data_long_filtered[, hover_text := paste0(
      "Date: ", format(PROD_DATE, "%b %Y"), "<br>",
      gsub("OilTrueRateBBLD", "Oil Rate (BBL/day)",
           gsub("CondensateRateBBLD", "Cond. Rate (BBL/day)",
                gsub("GasRateMCFD", "Gas Rate (MCF/day)",
                     gsub("BOERateBBLD", "BOE Rate (BBL/day)", ProductRateType)))),
      ": ", round(Rate, 2)
    )]
    
    product_colors_map <- setNames(custom_palette[1:4],
                                   c("OilTrueRateBBLD", "CondensateRateBBLD", "GasRateMCFD", "BOERateBBLD"))
    
    present_product_rate_types <- unique(plot_data_long_filtered$ProductRateType)
    active_colors <- product_colors_map[names(product_colors_map) %in% present_product_rate_types]
    
    all_rate_labels <- c("OilTrueRateBBLD" = "Oil (BBL/day)",
                         "CondensateRateBBLD" = "Cond. (BBL/day)",
                         "GasRateMCFD" = "Gas (MCF/day)",
                         "BOERateBBLD" = "BOE (BBL/day)")
    active_labels <- all_rate_labels[names(all_rate_labels) %in% present_product_rate_types]
    
    if (is.factor(plot_data_long_filtered$ProductRateType)) {
      plot_data_long_filtered[, ProductRateType := factor(ProductRateType, levels = present_product_rate_types)]
    } else {
      plot_data_long_filtered[, ProductRateType := factor(ProductRateType, levels = present_product_rate_types)]
    }
    
    p <- ggplot(plot_data_long_filtered, aes(x = PROD_DATE, y = Rate, color = ProductRateType, group = ProductRateType, text = hover_text)) +
      geom_line(linewidth = 1) + geom_point(size = 1.5) +
      scale_y_continuous(labels = scales::comma) +
      scale_x_date(date_labels = "%b %Y", date_breaks = "1 year") +
      labs(title = plot_main_title, x = "Date", y = "Daily Rate", color = "Product Type") +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top") +
      scale_color_manual(values = active_colors, labels = active_labels, name = "Product Type")
    ggplotly(p, tooltip = "text")
  })
  
  dt_rowCallback_js <- JS(
    "function(row, data, displayNum, displayIndex, dataIndex) {",
    "  $('td', row).each(function() {",
    "    $(this).attr('title', $(this).text());",
    "  });",
    "}"
  )
  
  output$production_table <- DT::renderDataTable({
    table_data_for_well <- processed_production_for_plotting()
    if (is.null(table_data_for_well) || nrow(table_data_for_well) == 0) {
      current_uwi <- reactive_vals$current_selected_gsl_uwi_std
      msg <- if (!is.null(current_uwi) && current_uwi != "") paste("No production data for UWI:", current_uwi, "in selected date range or for selected product(s).")
      else "No well selected or no production data for selected product(s)."
      return(DT::datatable(data.frame(Message = msg), options = list(searching = FALSE, paging = FALSE, info=FALSE), rownames=FALSE))
    }
    cols_to_select <- c("PROD_DATE", "MonthlyOilTrueBBL", "MonthlyCondensateBBL", "MonthlyGasMCF",
                        "OilTrueRateBBLD", "CondensateRateBBLD", "GasRateMCFD", "BOERateBBLD",
                        "CumOilTrueBBL", "CumCondensateBBL", "CumGasMCF")
    actual_cols_present <- intersect(cols_to_select, names(table_data_for_well))
    if(length(actual_cols_present) == 0) {
      return(DT::datatable(data.frame(Message = "Required production columns not found."), options = list(searching = FALSE, paging = FALSE, info=FALSE), rownames=FALSE))
    }
    display_data <- table_data_for_well[, ..actual_cols_present]
    setnames(display_data, old = "PROD_DATE", new = "Prod. Month", skip_absent = TRUE)
    setnames(display_data, old = "MonthlyOilTrueBBL", new = "Oil (BBL/Month)", skip_absent = TRUE)
    setnames(display_data, old = "MonthlyCondensateBBL", new = "Cond. (BBL/Month)", skip_absent = TRUE)
    setnames(display_data, old = "MonthlyGasMCF", new = "Gas (MCF/Month)", skip_absent = TRUE)
    setnames(display_data, old = "OilTrueRateBBLD", new = "Oil Rate (BBL/day)", skip_absent = TRUE)
    setnames(display_data, old = "CondensateRateBBLD", new = "Cond. Rate (BBL/day)", skip_absent = TRUE)
    setnames(display_data, old = "GasRateMCFD", new = "Gas Rate (MCF/day)", skip_absent = TRUE)
    setnames(display_data, old = "BOERateBBLD", new = "BOE Rate (BBL/day)", skip_absent = TRUE)
    setnames(display_data, old = "CumOilTrueBBL", new = "Cum. Oil (BBL)", skip_absent = TRUE)
    setnames(display_data, old = "CumCondensateBBL", new = "Cum. Cond. (BBL)", skip_absent = TRUE)
    setnames(display_data, old = "CumGasMCF", new = "Cum. Gas (MCF)", skip_absent = TRUE)
    if ("Prod. Month" %in% names(display_data)) {
      if(inherits(display_data[['Prod. Month']], "Date")){ # Check if it's a Date
        display_data[, `Prod. Month` := format(as.Date(`Prod. Month`), "%Y-%m")]
      } else {
        # Attempt to convert if character
        tryCatch({
          display_data[, `Prod. Month` := format(as.Date(as.character(`Prod. Month`)), "%Y-%m")]
        }, error = function(e) {
          message(paste("Could not format Prod. Month for DT in production_table:", e$message))
        })
      }
    }
    rate_cols_to_format <- c("Oil Rate (BBL/day)", "Cond. Rate (BBL/day)", "Gas Rate (MCF/day)", "BOE Rate (BBL/day)")
    for(rc in rate_cols_to_format){
      if(rc %in% names(display_data)) display_data[[rc]] <- round(display_data[[rc]], 2)
    }
    DT::datatable(
      display_data,
      options = list(
        pageLength = 12,
        scrollX = TRUE,
        order = list(0, 'desc'),
        autoWidth = TRUE,
        columnDefs = list(list(className = 'dt-right', targets = which(sapply(display_data, is.numeric))-1 )),
        rowCallback = dt_rowCallback_js
      ),
      rownames = FALSE,
      caption = htmltools::tags$caption(style = "caption-side: top; text-align: center; font-weight:bold;", paste("Monthly, Daily Rate, & Cumulative Production Data for GSL_UWI:", reactive_vals$current_selected_gsl_uwi_std))
    )
  })
  
  output$download_prod_data <- downloadHandler(
    filename = function() {
      paste0("production_data_", reactive_vals$current_selected_gsl_uwi_std, "_", Sys.Date(), ".csv")
    },
    content = function(file) {
      data_to_download <- processed_production_for_plotting()
      if (nrow(data_to_download) > 0) {
        setnames(data_to_download, old = "PROD_DATE", new = "Prod_Month", skip_absent = TRUE)
        setnames(data_to_download, old = "MonthlyOilTrueBBL", new = "Monthly_Oil_BBL", skip_absent = TRUE)
        setnames(data_to_download, old = "MonthlyCondensateBBL", new = "Monthly_Condensate_BBL", skip_absent = TRUE)
        setnames(data_to_download, old = "MonthlyGasMCF", new = "Monthly_Gas_MCF", skip_absent = TRUE)
        setnames(data_to_download, old = "OilTrueRateBBLD", new = "Daily_Oil_Rate_BBLD", skip_absent = TRUE)
        setnames(data_to_download, old = "CondensateRateBBLD", new = "Daily_Condensate_Rate_BBLD", skip_absent = TRUE)
        setnames(data_to_download, old = "GasRateMCFD", new = "Daily_Gas_Rate_MCFD", skip_absent = TRUE)
        setnames(data_to_download, old = "BOERateBBLD", new = "Daily_BOE_Rate_BBLD", skip_absent = TRUE)
        setnames(data_to_download, old = "CumOilTrueBBL", new = "Cumulative_Oil_BBL", skip_absent = TRUE)
        setnames(data_to_download, old = "CumCondensateBBL", new = "Cumulative_Condensate_BBL", skip_absent = TRUE)
        setnames(data_to_download, old = "CumGasMCF", new = "Cumulative_Gas_MCF", skip_absent = TRUE)
        
        cols_for_download <- c("Prod_Month", "Monthly_Oil_BBL", "Monthly_Condensate_BBL", "Monthly_Gas_MCF",
                               "Daily_Oil_Rate_BBLD", "Daily_Condensate_Rate_BBLD", "Daily_Gas_Rate_MCFD", "Daily_BOE_Rate_BBLD",
                               "Cumulative_Oil_BBL", "Cumulative_Condensate_BBL", "Cumulative_Gas_MCF")
        data_to_download_final <- data_to_download[, .SD, .SDcols = intersect(cols_for_download, names(data_to_download))]
        fwrite(data_to_download_final, file)
      } else {
        fwrite(data.table(Message = "No production data available for selected well and date range."), file)
      }
    }
  )
  
  # --- Operator Group Cumulative (Separate from main filters) ---
  operator_group_prod_data <- eventReactive(input$update_group_plot, {
    req(wells_sf_global, input$group_operator_filter, input$group_prod_date_range, input$product_type_filter_analysis)
    selected_operators <- input$group_operator_filter
    selected_operators <- selected_operators[selected_operators != "" & selected_operators != "Loading..." & selected_operators != "No Operators Found" & selected_operators != "No Well Data"]
    if (is.null(selected_operators) || length(selected_operators) == 0) { showNotification("Please select at least one operator for operator group analysis.", type = "warning"); return(NULL) }
    
    selected_analysis_products <- input$product_type_filter_analysis
    if (is.null(selected_analysis_products) || length(selected_analysis_products) == 0) {
      showNotification("Please select at least one product type for analysis.", type = "warning")
      return(NULL)
    }
    
    showNotification("Fetching and processing operator group production data...", type = "message", duration = NULL, id="groupProdMsgOp")
    date_start <- input$group_prod_date_range[1]; date_end <- input$group_prod_date_range[2]
    target_uwis_dt <- as.data.table(wells_sf_global)[OperatorName %in% selected_operators, .(GSL_UWI_Std, OperatorName)]; target_uwis <- unique(target_uwis_dt$GSL_UWI_Std)
    if (length(target_uwis) == 0) { removeNotification("groupProdMsgOp"); showNotification("No wells found for the selected operator(s) in operator group analysis.", type = "warning"); return(NULL) }
    if (is.null(con) || !dbIsValid(con)) { con <<- connect_to_db(); if (is.null(con) || !dbIsValid(con)) { removeNotification("groupProdMsgOp"); return(NULL) } }
    uwi_batches <- split(target_uwis, ceiling(seq_along(target_uwis)/300)); all_prod_data_list <- list()
    for(batch_num in seq_along(uwi_batches)){
      current_batch_uwis <- uwi_batches[[batch_num]]
      sql_group_prod <- glue::glue_sql( "SELECT GSL_UWI, YEAR, PRODUCT_TYPE, ACTIVITY_TYPE, JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME, MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, SEP_VOLUME, OCT_VOLUME, NOV_VOLUME, DEC_VOLUME FROM PDEN_VOL_BY_MONTH WHERE GSL_UWI IN ({uwis*}) AND ACTIVITY_TYPE = 'PRODUCTION' AND PRODUCT_TYPE IN ('OIL', 'CND', 'GAS')", uwis = current_batch_uwis, .con = con)
      batch_prod_raw <- tryCatch({ data.table::as.data.table(dbGetQuery(con, sql_group_prod)) }, error = function(e) { data.table() }); if(nrow(batch_prod_raw) > 0) all_prod_data_list[[length(all_prod_data_list) + 1]] <- batch_prod_raw
    }
    if(length(all_prod_data_list) == 0){ removeNotification("groupProdMsgOp"); showNotification("No production data found for any wells in the selected operator group(s).", type = "warning"); return(NULL) }
    group_prod_raw <- rbindlist(all_prod_data_list, use.names = TRUE, fill = TRUE)
    
    # Filter by selected product types for analysis BEFORE cleaning and merging
    if (!is.null(selected_analysis_products) && length(selected_analysis_products) > 0 && !"BOE" %in% selected_analysis_products) { # If BOE is selected, we need all for BOE calc
      group_prod_raw <- group_prod_raw[toupper(PRODUCT_TYPE) %in% toupper(selected_analysis_products)]
    }
    if(nrow(group_prod_raw) == 0) {
      removeNotification("groupProdMsgOp");
      showNotification("No production data for the selected product type(s) in the operator group.", type = "warning");
      return(NULL)
    }
    
    if("GSL_UWI" %in% names(group_prod_raw)) { group_prod_raw[, GSL_UWI_Std_from_query := standardize_uwi(GSL_UWI)] }
    group_prod_cleaned <- clean_df_colnames(group_prod_raw, "Operator Group PDEN Volumes")
    if ("GSL_UWI_STD_FROM_QUERY" %in% names(group_prod_cleaned)) { setnames(group_prod_cleaned, "GSL_UWI_STD_FROM_QUERY", "GSL_UWI_Std") } else if ("GSL_UWI" %in% names(group_prod_cleaned) && !"GSL_UWI_Std" %in% names(group_prod_cleaned)) { group_prod_cleaned[, GSL_UWI_Std := standardize_uwi(GSL_UWI)]; group_prod_cleaned[, GSL_UWI := NULL] }
    if(!"GSL_UWI_Std" %in% names(target_uwis_dt)) { if("GSL_UWI" %in% names(target_uwis_dt)) { target_uwis_dt[, GSL_UWI_Std := standardize_uwi(GSL_UWI)] } else { removeNotification("groupProdMsgOp"); return(NULL) } }
    group_prod_cleaned <- merge(group_prod_cleaned, unique(target_uwis_dt[,.(GSL_UWI_Std, OperatorName)]), by="GSL_UWI_Std", all.x=TRUE); group_prod_cleaned <- group_prod_cleaned[!is.na(OperatorName)]
    if(nrow(group_prod_cleaned) == 0) { removeNotification("groupProdMsgOp"); return(NULL) }
    year_col_actual <- "YEAR"; product_type_col_actual <- "PRODUCT_TYPE"; oil_prod_val <- "OIL"; cnd_prod_val <- "CND"; gas_prod_val <- "GAS"
    month_abbrs_upper <- toupper(month.abb); monthly_vol_cols <- character(0); prod_dt_colnames_upper <- names(group_prod_cleaned)
    for(m_abbr in month_abbrs_upper){ potential_col_upper <- paste0(m_abbr, "_VOLUME"); if(potential_col_upper %in% prod_dt_colnames_upper) { monthly_vol_cols <- c(monthly_vol_cols, potential_col_upper) } }
    required_id_cols_group <- c("GSL_UWI_Std", "OperatorName", year_col_actual, product_type_col_actual)
    if(!(length(monthly_vol_cols) == 12 && all(required_id_cols_group %in% names(group_prod_cleaned)))) { removeNotification("groupProdMsgOp"); return(NULL) }
    for(col_name in monthly_vol_cols) { if (!is.numeric(group_prod_cleaned[[col_name]])) group_prod_cleaned[, (col_name) := as.numeric(get(col_name))] }
    if(!is.numeric(group_prod_cleaned[[year_col_actual]])) group_prod_cleaned[, (year_col_actual) := as.numeric(get(year_col_actual))]
    if(!is.character(group_prod_cleaned[[product_type_col_actual]])) group_prod_cleaned[, (product_type_col_actual) := as.character(get(product_type_col_actual))]
    group_prod_long <- data.table::melt(group_prod_cleaned, id.vars = required_id_cols_group, measure.vars = monthly_vol_cols, variable.name = "MONTH_VOLUME_COL", value.name = "Volume", na.rm = FALSE, verbose = FALSE)
    
    # Unit conversion based on the *original* PRODUCT_TYPE before aggregation
    group_prod_long[toupper(get(product_type_col_actual)) %in% c(toupper(oil_prod_val), toupper(cnd_prod_val)), Volume_Converted := Volume * M3_TO_BBL]
    group_prod_long[toupper(get(product_type_col_actual)) == toupper(gas_prod_val), Volume_Converted := Volume * E3M3_TO_MCF]
    group_prod_long[is.na(Volume_Converted), Volume_Converted := 0]
    
    group_prod_long <- group_prod_long[!is.na(Volume_Converted) & Volume_Converted != 0]
    if(nrow(group_prod_long) == 0) { removeNotification("groupProdMsgOp"); return(NULL) }
    
    group_prod_long[, Month_Num := match(toupper(substr(MONTH_VOLUME_COL, 1, 3)), month_abbrs_upper)]
    group_prod_long <- group_prod_long[!is.na(get(year_col_actual)) & !is.na(Month_Num)]
    group_prod_long[, PROD_DATE := tryCatch(as.Date(paste(get(year_col_actual), Month_Num, 1, sep="-"), format="%Y-%m-%d"), error = function(e) as.Date(NA))]
    group_prod_long <- group_prod_long[!is.na(PROD_DATE)]; group_prod_long <- group_prod_long[PROD_DATE >= date_start & PROD_DATE <= date_end]
    if(nrow(group_prod_long) == 0) { removeNotification("groupProdMsgOp"); return(NULL) }
    
    # Create specific product columns based on original type, then sum based on filter
    group_prod_long[, MonthlyOilTrueBBL_raw := fifelse(toupper(get(product_type_col_actual)) == toupper(oil_prod_val), Volume_Converted, 0)]
    group_prod_long[, MonthlyCondensateBBL_raw := fifelse(toupper(get(product_type_col_actual)) == toupper(cnd_prod_val), Volume_Converted, 0)]
    group_prod_long[, MonthlyGasMCF_raw := fifelse(toupper(get(product_type_col_actual)) == toupper(gas_prod_val), Volume_Converted, 0)]
    
    agg_by_operator_month <- group_prod_long[, .(
      MonthlyOilBBL = if("OIL" %in% selected_analysis_products || "BOE" %in% selected_analysis_products) sum(MonthlyOilTrueBBL_raw, na.rm = TRUE) else 0,
      MonthlyCondensateBBL = if("CND" %in% selected_analysis_products || "BOE" %in% selected_analysis_products) sum(MonthlyCondensateBBL_raw, na.rm = TRUE) else 0,
      MonthlyGasMCF = if("GAS" %in% selected_analysis_products || "BOE" %in% selected_analysis_products) sum(MonthlyGasMCF_raw, na.rm = TRUE) else 0
    ), by = .(OperatorName, PROD_DATE)][order(OperatorName, PROD_DATE)]
    
    # Ensure all selected products are actually present for BOE calculation
    oil_for_boe <- if("OIL" %in% selected_analysis_products || "BOE" %in% selected_analysis_products) agg_by_operator_month$MonthlyOilBBL else 0
    cnd_for_boe <- if("CND" %in% selected_analysis_products || "BOE" %in% selected_analysis_products) agg_by_operator_month$MonthlyCondensateBBL else 0
    gas_for_boe <- if("GAS" %in% selected_analysis_products || "BOE" %in% selected_analysis_products) agg_by_operator_month$MonthlyGasMCF else 0
    
    agg_by_operator_month[, DaysInMonth := lubridate::days_in_month(PROD_DATE)]
    agg_by_operator_month[, AvgOilRateBBLD := MonthlyOilBBL / DaysInMonth] # This is now sum of selected OIL + CND
    agg_by_operator_month[, AvgGasRateMCFD := MonthlyGasMCF / DaysInMonth]
    agg_by_operator_month[, AvgBOERateBBLD := (oil_for_boe + cnd_for_boe + (gas_for_boe / MCF_PER_BOE)) / DaysInMonth]
    
    agg_by_operator_month[, CumOilBBL := cumsum(MonthlyOilBBL), by = .(OperatorName)] # This is cum of selected OIL+CND
    agg_by_operator_month[, CumGasMCF := cumsum(MonthlyGasMCF), by = .(OperatorName)]
    removeNotification("groupProdMsgOp"); showNotification("Operator group production data processed.", type="message"); return(agg_by_operator_month)
  })
  output$grouped_cumulative_plot <- renderPlot({
    plot_data <- operator_group_prod_data()
    req(plot_data); if(nrow(plot_data) == 0) { return(ggplot() + labs(title = "No data to display for selected operator group(s) and date range.", x=NULL, y=NULL) + theme_void()) }
    
    plot_data_long <- data.table::melt(plot_data, id.vars = c("OperatorName", "PROD_DATE"),
                                       measure.vars = c("AvgOilRateBBLD", "AvgGasRateMCFD", "AvgBOERateBBLD"),
                                       variable.name = "ProductRateType", value.name = "AverageDailyRate")
    plot_data_long_filtered <- plot_data_long[AverageDailyRate > 0 & is.finite(AverageDailyRate)]
    if(nrow(plot_data_long_filtered) == 0){ return(ggplot() + labs(title = "No average daily rates to display for operator group.", x="Date", y="Average Daily Rate") + theme_minimal()) }
    
    # Determine number of unique operators for color palette
    unique_ops_plot <- unique(plot_data_long_filtered$OperatorName)
    operator_colors <- custom_palette[1:min(length(unique_ops_plot), length(custom_palette))]
    if (length(unique_ops_plot) > length(custom_palette)) { # Recycle if more ops than colors
      operator_colors <- rep(custom_palette, length.out = length(unique_ops_plot))
    }
    names(operator_colors) <- unique_ops_plot
    
    
    rate_labels_group <- c("AvgOilRateBBLD" = "Avg Oil/Cond (BBL/day)", "AvgGasRateMCFD" = "Avg Gas (MCF/day)", "AvgBOERateBBLD" = "Avg BOE (BBL/day)")
    
    ggplot(plot_data_long_filtered, aes(x = PROD_DATE, y = AverageDailyRate, color = OperatorName, linetype = ProductRateType, group = interaction(OperatorName, ProductRateType))) +
      geom_line(linewidth = 1.2) +
      scale_y_continuous(labels = scales::comma) +
      scale_x_date(date_labels = "%b %Y", date_breaks = "1 year") +
      labs(title = "Operator Group Average Daily Production Rate", x = "Date", y = "Average Daily Rate", color = "Operator", linetype = "Product Type") +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top") +
      scale_color_manual(values = operator_colors) + # Apply custom colors
      scale_linetype_manual(values = c("AvgOilRateBBLD" = "solid", "AvgGasRateMCFD" = "dashed", "AvgBOERateBBLD" = "dotted"),
                            labels = rate_labels_group) +
      guides(color = guide_legend(override.aes = list(linetype = "solid")))
  })
  output$grouped_production_table <- DT::renderDataTable({
    table_data <- operator_group_prod_data()
    if (is.null(table_data) || nrow(table_data) == 0) { return(DT::datatable(data.frame(Message = "No operator group production data to display."), options = list(searching = FALSE, paging = FALSE, info=FALSE), rownames=FALSE)) }
    display_data <- copy(table_data)
    setnames(display_data, old=c("PROD_DATE", "OperatorName", "MonthlyOilBBL", "MonthlyGasMCF", "AvgOilRateBBLD", "AvgGasRateMCFD", "AvgBOERateBBLD", "CumOilBBL", "CumGasMCF"),
             new=c("Prod. Month", "Operator", "Monthly Oil/Cond (BBL)", "Monthly Gas (MCF)", "Avg Oil/Cond Rate (BBL/day)", "Avg Gas Rate (MCF/day)", "Avg BOE Rate (BBL/day)", "Cum. Oil/Cond (BBL)", "Cum. Gas (MCF)"), skip_absent = TRUE)
    if ("Prod. Month" %in% names(display_data) && inherits(display_data[['Prod. Month']], "Date")) {
      display_data[, `Prod. Month` := format(as.Date(`Prod. Month`), "%Y-%m")]
    }
    rate_cols_to_format_group <- c("Avg Oil/Cond Rate (BBL/day)", "Avg Gas Rate (MCF/day)", "Avg BOE Rate (BBL/day)")
    for(rcg in rate_cols_to_format_group){ if(rcg %in% names(display_data)) display_data[[rcg]] <- round(display_data[[rcg]], 2) }
    DT::datatable(
      display_data,
      options = list(
        pageLength = 10,
        scrollX = TRUE,
        order = list(list(1, 'asc'), list(0, 'desc')),
        rowCallback = dt_rowCallback_js
      ),
      rownames = FALSE,
      caption = "Operator Group Monthly, Avg Daily Rate, and Cumulative Production"
    )
  })
  output$download_group_prod_data <- downloadHandler(
    filename = function() { paste0("operator_group_production_summary_", Sys.Date(), ".csv") },
    content = function(file) {
      data_to_download <- operator_group_prod_data()
      if (!is.null(data_to_download) && nrow(data_to_download) > 0) {
        setnames(data_to_download, old=c("PROD_DATE", "OperatorName", "MonthlyOilBBL", "MonthlyGasMCF", "AvgOilRateBBLD", "AvgGasRateMCFD", "AvgBOERateBBLD", "CumOilBBL", "CumGasMCF"),
                 new=c("Prod_Month", "Operator", "Monthly_Oil_Cond_BBL", "Monthly_Gas_MCF", "Avg_Daily_Oil_Cond_Rate_BBLD", "Avg_Daily_Gas_Rate_MCFD", "Avg_Daily_BOE_Rate_BBLD", "Cumulative_Oil_Cond_BBL", "Cumulative_Gas_MCF"), skip_absent = TRUE)
        fwrite(data_to_download, file)
      } else { fwrite(data.table(Message = "No operator group production data available for download."), file) }
    }
  )
  
  filtered_group_cumulative_data <- eventReactive(input$calculate_filtered_cumulative, {
    req(wells_sf_global, input$product_type_filter_analysis, input$filtered_group_breakout_by)
    
    selected_analysis_products <- input$product_type_filter_analysis
    if (is.null(selected_analysis_products) || length(selected_analysis_products) == 0) {
      showNotification("Please select at least one product type for analysis (under 'Production Analysis' main tab).", type = "warning", duration=5)
      return(NULL)
    }
    
    showNotification("Calculating normalized rate for filtered wells by operator...", type = "message", duration = NULL, id="filtRateMsg")
    filtered_wells_sf_for_calc <- reactive_vals$wells_to_display
    if (is.null(filtered_wells_sf_for_calc) || nrow(filtered_wells_sf_for_calc) == 0) { removeNotification("filtRateMsg"); showNotification("No wells selected by current map filters.", type = "warning"); return(NULL) }
    
    if (!"LateralLength" %in% names(filtered_wells_sf_for_calc)) {
      removeNotification("filtRateMsg");
      showNotification("LateralLength column not found in well data. Cannot normalize.", type = "error", duration=5);
      return(NULL)
    }
    breakout_cols_needed <- c("GSL_UWI_Std", "OperatorName", "LateralLength", "Formation", "FieldName", "ProvinceState", "FirstProdDate")
    breakout_cols_needed_present <- breakout_cols_needed[breakout_cols_needed %in% names(filtered_wells_sf_for_calc)]
    
    
    target_uwis_with_breakout_cols <- unique(as.data.table(sf::st_drop_geometry(filtered_wells_sf_for_calc))[, ..breakout_cols_needed_present])
    
    if ("FirstProdDate" %in% names(target_uwis_with_breakout_cols) && inherits(target_uwis_with_breakout_cols$FirstProdDate, "Date")) {
      target_uwis_with_breakout_cols[, FirstProdYear := as.character(year(FirstProdDate))]
    } else {
      target_uwis_with_breakout_cols[, FirstProdYear := NA_character_]
    }
    
    
    target_uwis <- target_uwis_with_breakout_cols$GSL_UWI_Std
    target_uwis <- target_uwis[!is.na(target_uwis) & target_uwis != ""]
    
    if (length(target_uwis) == 0) { removeNotification("filtRateMsg"); showNotification("No valid GSL_UWI_Std found for filtered wells.", type = "warning"); return(NULL) }
    if (is.null(con) || !dbIsValid(con)) { con <<- connect_to_db(); if (is.null(con) || !dbIsValid(con)) { removeNotification("filtRateMsg"); return(NULL) } }
    
    uwi_batches <- split(target_uwis, ceiling(seq_along(target_uwis)/300)); all_prod_data_list <- list()
    for(batch_num in seq_along(uwi_batches)){
      current_batch_uwis <- uwi_batches[[batch_num]]
      sql_prod <- glue::glue_sql( "SELECT GSL_UWI, YEAR, PRODUCT_TYPE, ACTIVITY_TYPE, JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME, MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, SEP_VOLUME, OCT_VOLUME, NOV_VOLUME, DEC_VOLUME FROM PDEN_VOL_BY_MONTH WHERE GSL_UWI IN ({uwis*}) AND ACTIVITY_TYPE = 'PRODUCTION' AND PRODUCT_TYPE IN ('OIL', 'CND', 'GAS')", uwis = current_batch_uwis, .con = con)
      batch_prod_raw <- tryCatch({ data.table::as.data.table(dbGetQuery(con, sql_prod)) }, error = function(e) { data.table()}); if(nrow(batch_prod_raw) > 0) all_prod_data_list[[length(all_prod_data_list) + 1]] <- batch_prod_raw
    }
    if(length(all_prod_data_list) == 0){ removeNotification("filtRateMsg"); showNotification("No production data found for filtered wells.", type = "warning"); return(NULL) }
    full_prod_raw <- rbindlist(all_prod_data_list, use.names = TRUE, fill = TRUE)
    
    # Filter by selected product types for analysis
    if (!("BOE" %in% selected_analysis_products) && !is.null(selected_analysis_products) && length(selected_analysis_products) > 0) {
      full_prod_raw <- full_prod_raw[toupper(PRODUCT_TYPE) %in% toupper(selected_analysis_products)]
    }
    if(nrow(full_prod_raw) == 0) {
      removeNotification("filtRateMsg");
      showNotification("No production data for the selected product type(s) in the filtered group.", type = "warning");
      return(NULL)
    }
    
    
    if("GSL_UWI" %in% names(full_prod_raw)) { full_prod_raw[, GSL_UWI_Std_query := standardize_uwi(GSL_UWI)] }
    prod_cleaned <- clean_df_colnames(full_prod_raw, "Filtered Group PDEN Volumes")
    
    if ("GSL_UWI_STD_QUERY" %in% names(prod_cleaned)) {
      setnames(prod_cleaned, "GSL_UWI_STD_QUERY", "GSL_UWI_Std")
    } else if ("GSL_UWI" %in% names(prod_cleaned) && !"GSL_UWI_Std" %in% names(prod_cleaned)) {
      prod_cleaned[, GSL_UWI_Std := standardize_uwi(GSL_UWI)]
    } else if (!"GSL_UWI_Std" %in% names(prod_cleaned)) {
      removeNotification("filtRateMsg"); return(NULL)
    }
    
    prod_cleaned <- merge(prod_cleaned, target_uwis_with_breakout_cols, by = "GSL_UWI_Std", all.x = TRUE, allow.cartesian=TRUE)
    
    breakout_col_name <- input$filtered_group_breakout_by
    if (!breakout_col_name %in% names(prod_cleaned)) {
      removeNotification("filtRateMsg"); showNotification(paste("Breakout column '", breakout_col_name, "' not found after merging production data."), type = "error"); return(NULL)
    }
    prod_cleaned <- prod_cleaned[!is.na(get(breakout_col_name))]
    
    if(nrow(prod_cleaned) == 0) { removeNotification("filtRateMsg"); showNotification(paste("No production data for valid groups based on:", breakout_col_name), type = "warning"); return(NULL) }
    
    year_col <- "YEAR"; product_col <- "PRODUCT_TYPE"
    oil_only_val <- "OIL"; cnd_only_val <- "CND"; gas_val <- "GAS"
    month_cols <- paste0(toupper(month.abb), "_VOLUME");
    
    # OperatorName must be present for calendar cumulative plot, even if not the breakout
    cols_to_check_base <- c(year_col, product_col, breakout_col_name, "LateralLength", "GSL_UWI_Std", "OperatorName")
    if(!all(cols_to_check_base %in% names(prod_cleaned)) || !all(month_cols %in% names(prod_cleaned))) {
      missing_cols <- c(cols_to_check_base[!cols_to_check_base %in% names(prod_cleaned)], month_cols[!month_cols %in% names(prod_cleaned)])
      warning(paste("Filtered Group: Missing required columns after cleaning:", paste(missing_cols, collapse=", ")))
      removeNotification("filtRateMsg"); return(NULL)
    }
    
    for(mc in month_cols) if(!is.numeric(prod_cleaned[[mc]])) prod_cleaned[, (mc) := as.numeric(get(mc))]
    if(!is.numeric(prod_cleaned[[year_col]])) prod_cleaned[, (year_col) := as.numeric(get(year_col))]
    if(!is.character(prod_cleaned[[product_col]])) prod_cleaned[, (product_col) := as.character(get(product_col))]
    
    id_vars_melt <- c("GSL_UWI_Std", year_col, product_col, breakout_col_name, "LateralLength", "OperatorName")
    id_vars_melt <- unique(id_vars_melt)
    prod_long <- melt(prod_cleaned, id.vars = id_vars_melt, measure.vars = month_cols, variable.name = "MONTH_VOL_COL", value.name = "Volume", na.rm = FALSE)
    
    prod_long[toupper(get(product_col)) %in% c(toupper(oil_only_val), toupper(cnd_only_val)), Volume_Converted := Volume * M3_TO_BBL]
    prod_long[toupper(get(product_col)) == toupper(gas_val), Volume_Converted := Volume * E3M3_TO_MCF]
    prod_long[is.na(Volume_Converted), Volume_Converted := 0]
    
    prod_long <- prod_long[!is.na(Volume_Converted) & Volume_Converted != 0]
    if(nrow(prod_long) == 0) { removeNotification("filtRateMsg"); return(NULL) }
    prod_long[, Month_Num := match(toupper(substr(MONTH_VOL_COL, 1, 3)), toupper(month.abb))]
    prod_long <- prod_long[!is.na(get(year_col)) & !is.na(Month_Num)]
    prod_long[, PROD_DATE := as.Date(paste(get(year_col), Month_Num, 1, sep="-"), format="%Y-%m-%d")]; prod_long <- prod_long[!is.na(PROD_DATE)]
    
    prod_long[, MonthlyOilTrueBBL_raw := fifelse(toupper(get(product_col)) == toupper(oil_only_val), Volume_Converted, 0)]
    prod_long[, MonthlyCondensateBBL_raw := fifelse(toupper(get(product_col)) == toupper(cnd_only_val), Volume_Converted, 0)]
    prod_long[, MonthlyGasMCF_raw := fifelse(toupper(get(product_col)) == toupper(gas_val), Volume_Converted, 0)]
    
    prod_long[, DaysInMonth := lubridate::days_in_month(PROD_DATE)]
    prod_long[, DailyBOE_well := (MonthlyOilTrueBBL_raw + MonthlyCondensateBBL_raw + (MonthlyGasMCF_raw / MCF_PER_BOE)) / DaysInMonth]
    prod_long[, DailyBOE_per_1000ft_well := ifelse(LateralLength > 0 & !is.na(LateralLength), DailyBOE_well / (LateralLength / 1000), NA_real_)]
    
    prod_long[, PeakDailyBOE_well := max(DailyBOE_well, na.rm = TRUE), by = GSL_UWI_Std]
    peak_info_well <- prod_long[DailyBOE_well > 0 & DailyBOE_well == PeakDailyBOE_well, .(PeakProdDate_well = min(PROD_DATE)), by = GSL_UWI_Std]
    
    prod_long_from_peak <- merge(prod_long, peak_info_well, by = "GSL_UWI_Std", all.x = TRUE)
    prod_long_from_peak <- prod_long_from_peak[!is.na(PeakProdDate_well) & PROD_DATE >= PeakProdDate_well]
    if(nrow(prod_long_from_peak) == 0) { removeNotification("filtRateMsg"); return(NULL) }
    
    prod_long_from_peak[, MonthOnProd := round(as.numeric(PROD_DATE - PeakProdDate_well) / AVG_DAYS_PER_MONTH) + 1]
    
    grouping_vars_norm <- c(breakout_col_name, "MonthOnProd")
    agg_by_month_group_norm <- prod_long_from_peak[, .(
      AvgNormBOERate_per_1000ft = mean(DailyBOE_per_1000ft_well, na.rm = TRUE),
      WellCount_Norm = uniqueN(GSL_UWI_Std)
    ), by = grouping_vars_norm][order(get(breakout_col_name), MonthOnProd)]
    
    agg_by_month_group_norm[is.na(AvgNormBOERate_per_1000ft), AvgNormBOERate_per_1000ft := 0]
    agg_by_month_group_norm[, CumNormBOE_per_1000ft := cumsum(AvgNormBOERate_per_1000ft * AVG_DAYS_PER_MONTH), by = c(breakout_col_name)]
    
    # --- MODIFIED: Calendar plot data now includes daily rate ---
    agg_by_calendar_month_group <- prod_long[, .(
      TotalMonthlyBOE_Cal = sum(MonthlyOilTrueBBL_raw + MonthlyCondensateBBL_raw + (MonthlyGasMCF_raw / MCF_PER_BOE), na.rm = TRUE)
    ), by = c("PROD_DATE", breakout_col_name)][order(get(breakout_col_name), PROD_DATE)]
    agg_by_calendar_month_group[, DaysInMonth := lubridate::days_in_month(PROD_DATE)]
    agg_by_calendar_month_group[, AvgDailyBOERate_Cal := TotalMonthlyBOE_Cal / DaysInMonth]
    agg_by_calendar_month_group[, CumBOE_Calendar := cumsum(TotalMonthlyBOE_Cal), by = c(breakout_col_name)]
    # --- END MODIFICATION ---
    
    # Table data: Aggregate on calendar basis, grouped by the selected breakout column
    table_data_prep <- prod_long[, .(
      TotalMonthlyOilTrueBBL_sum = sum(MonthlyOilTrueBBL_raw, na.rm = TRUE), # Sum for the group-month
      TotalMonthlyCondensateBBL_sum = sum(MonthlyCondensateBBL_raw, na.rm = TRUE),
      TotalMonthlyGasMCF_sum = sum(MonthlyGasMCF_raw, na.rm = TRUE),
      DailyBOE_per_1000ft_well_avg = mean(DailyBOE_per_1000ft_well, na.rm = TRUE), # Average of per-well normalized rates
      DailyBOE_well_avg = mean(DailyBOE_well, na.rm=TRUE), # Average of per-well BOE rates
      TotalLateralLength_sum = sum(fifelse(DailyBOE_well > 0 & !is.na(LateralLength), LateralLength, 0), na.rm=TRUE) # Sum LL of producing wells in group-month
    ), by = c("PROD_DATE", breakout_col_name)]
    
    table_data_final <- table_data_prep %>%
      .[, DaysInMonth := lubridate::days_in_month(PROD_DATE)] %>%
      .[, TotalMonthlyBOE := (DailyBOE_well_avg * DaysInMonth)] %>% # Recalculate total from avg daily
      .[order(get(breakout_col_name), PROD_DATE)] %>%
      .[, CumTotalMonthlyBOE := cumsum(TotalMonthlyBOE), by = c(breakout_col_name)] %>%
      setnames(old = c("DailyBOE_per_1000ft_well_avg", "DailyBOE_well_avg", "TotalSumLateralLength_sum"),
               new = c("AvgDailyBOE_per_1000ft", "AvgDailyBOE", "TotalSumLateralLength"), skip_absent = TRUE)
    
    
    removeNotification("filtRateMsg"); showNotification("Filtered group normalized rates processed.", type="message");
    return(list(
      normalized_data = agg_by_month_group_norm,
      calendar_cumulative_data = agg_by_calendar_month_group,
      table_data = table_data_final
    ))
  })
  output$filtered_group_plot_title_normalized <- renderUI({
    breakout_choice_map <- c("OperatorName" = "Operator",
                             "Formation" = "Formation",
                             "FieldName" = "Field",
                             "ProvinceState" = "Province/State",
                             "FirstProdYear" = "First Prod Year")
    display_breakout_name <- breakout_choice_map[input$filtered_group_breakout_by]
    if(is.na(display_breakout_name) || is.null(display_breakout_name)) display_breakout_name <- input$filtered_group_breakout_by
    h5(paste("Average Daily BOE Rate per 1000ft Lateral by", display_breakout_name, "(Time Normalized)"))
  })
  output$filtered_group_plot_title_cumulative_boe <- renderUI({
    breakout_choice_map <- c("OperatorName" = "Operator",
                             "Formation" = "Formation",
                             "FieldName" = "Field",
                             "ProvinceState" = "Province/State",
                             "FirstProdYear" = "First Prod Year")
    display_breakout_name <- breakout_choice_map[input$filtered_group_breakout_by]
    if(is.na(display_breakout_name) || is.null(display_breakout_name)) display_breakout_name <- input$filtered_group_breakout_by
    h5(paste("Cumulative BOE Production by", display_breakout_name, "(Calendar Time)"))
  })
  
  output$filtered_group_cumulative_plot_normalized <- renderPlot({
    analysis_data <- filtered_group_cumulative_data()
    req(analysis_data, analysis_data$normalized_data)
    plot_data <- analysis_data$normalized_data
    
    if(is.null(plot_data) || nrow(plot_data) == 0) { return(ggplot() + labs(title = "No normalized rate data for filtered wells.", x=NULL, y=NULL) + theme_void()) }
    
    plot_data_for_ggplot <- plot_data[!is.na(AvgNormBOERate_per_1000ft) & is.finite(AvgNormBOERate_per_1000ft) & AvgNormBOERate_per_1000ft >= 0] # Allow 0 for plotting start
    if(nrow(plot_data_for_ggplot) == 0) { return(ggplot() + labs(title="No positive normalized BOE rates for filtered wells by selected group.", x="Month on Production", y="Normalized BOE Rate (BOE/d per 1000ft)") + theme_minimal()) }
    
    breakout_column_r_name <- input$filtered_group_breakout_by
    
    if (!breakout_column_r_name %in% names(plot_data_for_ggplot)) {
      warning(paste("Breakout column", breakout_column_r_name, "not found in normalized plot data for ggplot."))
      return(ggplot() + labs(title = paste("Error: Breakout column",input$filtered_group_breakout_by,"not found." )) + theme_void())
    }
    unique_groups_plot <- unique(plot_data_for_ggplot[[breakout_column_r_name]])
    
    group_colors_plot <- custom_palette[1:min(length(unique_groups_plot), length(custom_palette))]
    if (length(unique_groups_plot) > length(custom_palette)) {
      group_colors_plot <- rep(custom_palette, length.out = length(unique_groups_plot))
    }
    names(group_colors_plot) <- unique_groups_plot
    
    # Primary y-axis label
    y_lab_primary <- "Avg Daily BOE Rate per 1000m Lateral (Solid)"
    
    p_norm <- ggplot(plot_data_for_ggplot, aes(x = MonthOnProd, y = AvgNormBOERate_per_1000ft, color = .data[[breakout_column_r_name]], group = .data[[breakout_column_r_name]])) +
      geom_line(linewidth = 1.2) +
      scale_x_continuous(name = "Month on Production", breaks = scales::pretty_breaks(n=10)) +
      labs(color = names(which(c("OperatorName" = "Operator", "Formation" = "Formation", "FieldName" = "Field", "ProvinceState" = "Province/State", "FirstProdYear" = "First Prod Year") == breakout_column_r_name))) +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top") +
      scale_color_manual(values = group_colors_plot)
    
    # Add secondary axis for cumulative normalized production conditionally
    if ("CumNormBOE_per_1000ft" %in% names(plot_data_for_ggplot) && any(!is.na(plot_data_for_ggplot$CumNormBOE_per_1000ft) & plot_data_for_ggplot$CumNormBOE_per_1000ft > 0)) {
      max_rate <- max(plot_data_for_ggplot$AvgNormBOERate_per_1000ft, na.rm = TRUE)
      max_cum_data <- plot_data_for_ggplot[!is.na(CumNormBOE_per_1000ft)]
      max_cum <- if(nrow(max_cum_data) > 0) max(max_cum_data$CumNormBOE_per_1000ft, na.rm = TRUE) else 0
      
      if (is.finite(max_rate) && max_rate > 0 && is.finite(max_cum) && max_cum > 0) {
        scale_factor <- max_rate / max_cum
        if (is.finite(scale_factor) && scale_factor != 0) {
          p_norm <- p_norm +
            geom_line(data=plot_data_for_ggplot, aes(y = CumNormBOE_per_1000ft * scale_factor), linetype = "dashed", linewidth = 0.8) + # inherit.aes = FALSE removed to use existing color mapping
            scale_y_continuous( # Define both axes here
              name = y_lab_primary,
              labels = scales::comma,
              sec.axis = sec_axis(~ . / scale_factor, name = "Cum. Norm. BOE per 1000m (Dashed)", labels = scales::comma)
            )
        } else {
          p_norm <- p_norm + scale_y_continuous(name = y_lab_primary, labels = scales::comma)
        }
      } else {
        p_norm <- p_norm + scale_y_continuous(name = y_lab_primary, labels = scales::comma)
      }
    } else {
      p_norm <- p_norm + scale_y_continuous(name = y_lab_primary, labels = scales::comma)
    }
    return(p_norm)
  })
  output$filtered_group_cumulative_plot_cumulative_boe <- renderPlot({
    analysis_data <- filtered_group_cumulative_data()
    req(analysis_data, analysis_data$calendar_cumulative_data)
    plot_data <- analysis_data$calendar_cumulative_data # This now contains the breakout column
    
    breakout_column_r_name_cum <- input$filtered_group_breakout_by # Use the same breakout
    
    if(nrow(plot_data) == 0) { return(ggplot() + labs(title = "No cumulative BOE data for filtered wells.", x=NULL, y=NULL) + theme_void()) }
    
    plot_data_for_ggplot <- plot_data[!is.na(CumBOE_Calendar) & is.finite(CumBOE_Calendar)]
    if(nrow(plot_data_for_ggplot) == 0) { return(ggplot() + labs(title=paste("No cumulative BOE data for filtered wells by", breakout_column_r_name_cum), x="Date", y="Cumulative BOE") + theme_minimal()) }
    
    if (!breakout_column_r_name_cum %in% names(plot_data_for_ggplot)) {
      warning(paste("Breakout column", breakout_column_r_name_cum, "not found in calendar cumulative plot data for ggplot."))
      return(ggplot() + labs(title = paste("Error: Breakout column", breakout_column_r_name_cum,"not found." )) + theme_void())
    }
    unique_groups_plot_cum <- unique(plot_data_for_ggplot[[breakout_column_r_name_cum]])
    group_colors_plot_cum <- custom_palette[1:min(length(unique_groups_plot_cum), length(custom_palette))]
    if (length(unique_groups_plot_cum) > length(custom_palette)) {
      group_colors_plot_cum <- rep(custom_palette, length.out = length(unique_groups_plot_cum))
    }
    names(group_colors_plot_cum) <- unique_groups_plot_cum
    
    ggplot(plot_data_for_ggplot, aes(x = PROD_DATE, y = CumBOE_Calendar / 1000, color = .data[[breakout_column_r_name_cum]], group = .data[[breakout_column_r_name_cum]])) +
      geom_line(linewidth = 1.2) +
      scale_y_continuous(labels = scales::comma) +
      scale_x_date(date_labels = "%b %Y", date_breaks = "1 year") +
      labs(title = paste("Cumulative BOE Production by", names(which(c("OperatorName" = "Operator", "Formation" = "Formation", "FieldName" = "Field", "ProvinceState" = "Province/State", "FirstProdYear" = "First Prod Year") == breakout_column_r_name_cum)), "(Calendar Time)"),
           x = "Date", y = "Cumulative BOE (MBOE)", color = names(which(c("OperatorName" = "Operator", "Formation" = "Formation", "FieldName" = "Field", "ProvinceState" = "Province/State", "FirstProdYear" = "First Prod Year") == breakout_column_r_name_cum))) +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top") +
      scale_color_manual(values = group_colors_plot_cum)
  })
  
  # --- NEW PLOT TITLE AND RENDER LOGIC ---
  output$filtered_group_plot_title_calendar_rate <- renderUI({
    breakout_choice_map <- c("OperatorName" = "Operator",
                             "Formation" = "Formation",
                             "FieldName" = "Field",
                             "ProvinceState" = "Province/State",
                             "FirstProdYear" = "First Prod Year")
    display_breakout_name <- breakout_choice_map[input$filtered_group_breakout_by]
    if(is.na(display_breakout_name) || is.null(display_breakout_name)) display_breakout_name <- input$filtered_group_breakout_by
    h5(paste("Average Daily BOE Rate by", display_breakout_name, "(Calendar Time)"))
  })
  
  output$filtered_group_calendar_rate_plot <- renderPlot({
    analysis_data <- filtered_group_cumulative_data()
    req(analysis_data, analysis_data$calendar_cumulative_data)
    plot_data <- analysis_data$calendar_cumulative_data
    
    breakout_column_r_name_rate <- input$filtered_group_breakout_by
    
    if(nrow(plot_data) == 0 || !"AvgDailyBOERate_Cal" %in% names(plot_data)) {
      return(ggplot() + labs(title = "No calendar rate data for filtered wells.", x=NULL, y=NULL) + theme_void())
    }
    
    plot_data_for_ggplot <- plot_data[!is.na(AvgDailyBOERate_Cal) & is.finite(AvgDailyBOERate_Cal)]
    if(nrow(plot_data_for_ggplot) == 0) {
      return(ggplot() + labs(title=paste("No daily rate data for filtered wells by", breakout_column_r_name_rate), x="Date", y="Avg Daily BOE Rate") + theme_minimal())
    }
    
    if (!breakout_column_r_name_rate %in% names(plot_data_for_ggplot)) {
      warning(paste("Breakout column", breakout_column_r_name_rate, "not found in calendar rate plot data for ggplot."))
      return(ggplot() + labs(title = paste("Error: Breakout column", breakout_column_r_name_rate,"not found." )) + theme_void())
    }
    
    unique_groups_plot_rate <- unique(plot_data_for_ggplot[[breakout_column_r_name_rate]])
    group_colors_plot_rate <- custom_palette[1:min(length(unique_groups_plot_rate), length(custom_palette))]
    if (length(unique_groups_plot_rate) > length(custom_palette)) {
      group_colors_plot_rate <- rep(custom_palette, length.out = length(unique_groups_plot_rate))
    }
    names(group_colors_plot_rate) <- unique_groups_plot_rate
    
    legend_title <- names(which(c("OperatorName" = "Operator", "Formation" = "Formation", "FieldName" = "Field", "ProvinceState" = "Province/State", "FirstProdYear" = "First Prod Year") == breakout_column_r_name_rate))
    
    ggplot(plot_data_for_ggplot, aes(x = PROD_DATE, y = AvgDailyBOERate_Cal, color = .data[[breakout_column_r_name_rate]], group = .data[[breakout_column_r_name_rate]])) +
      geom_line(linewidth = 1.2) +
      scale_y_continuous(labels = scales::comma) +
      scale_x_date(date_labels = "%b %Y", date_breaks = "1 year") +
      labs(
        x = "Date", y = "Average Daily BOE Rate (boe/d)",
        color = legend_title
      ) +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top") +
      scale_color_manual(values = group_colors_plot_rate)
  })
  # --- END NEW PLOT LOGIC ---
  
  output$filtered_group_production_table <- DT::renderDataTable({
    analysis_data <- filtered_group_cumulative_data()
    if(is.null(analysis_data) || is.null(analysis_data$table_data) || nrow(analysis_data$table_data) == 0){
      return(DT::datatable(data.frame(Message = "No data for filtered group table. Check filters or click 'Calculate' button."),
                           options = list(searching=FALSE, paging=FALSE, info=FALSE), rownames=FALSE))
    }
    table_data <- analysis_data$table_data
    
    display_data <- copy(table_data)
    
    breakout_col_r_name <- input$filtered_group_breakout_by
    breakout_col_display_name <- names(which(c("OperatorName" = "Operator", "Formation" = "Formation", "FieldName" = "Field", "ProvinceState" = "Province/State", "FirstProdYear" = "First Prod Year") == breakout_col_r_name))
    if(length(breakout_col_display_name) == 0) breakout_col_display_name <- breakout_col_r_name
    
    # Rename columns for display, ensuring the dynamic breakout column is handled
    # The `table_data_final` should already have the breakout column with its original R name
    current_names <- names(display_data)
    new_names_map <- list(
      "PROD_DATE" = "Prod. Month",
      "TotalMonthlyBOE" = "Total Monthly BOE",
      "AvgDailyBOE" = "Avg Daily BOE",
      "AvgDailyBOE_per_1000ft" = "Avg Daily BOE/1kft",
      "CumTotalMonthlyBOE" = "Cum. BOE",
      "TotalSumLateralLength" = "Sum Prod. LatLen (ft)"
    )
    # Add the dynamic breakout column to the map
    new_names_map[[breakout_col_r_name]] <- breakout_col_display_name
    
    for(old_n in names(new_names_map)){
      if(old_n %in% current_names){
        setnames(display_data, old = old_n, new = new_names_map[[old_n]], skip_absent = TRUE)
      }
    }
    
    if ("Prod. Month" %in% names(display_data)) {
      # Ensure 'Prod. Month' is Date before formatting
      if(!inherits(display_data[['Prod. Month']], "Date")){
        display_data[, `Prod. Month` := tryCatch(as.Date(as.character(`Prod. Month`)), error = function(e) NA_Date_)]
      }
      display_data[!is.na(`Prod. Month`), `Prod. Month` := format(`Prod. Month`, "%Y-%m")]
    }
    
    numeric_cols_to_round <- c("Total Monthly BOE", "Avg Daily BOE", "Avg Daily BOE/1kft", "Cum. BOE", "Sum Prod. LatLen (ft)")
    for(rcfg in numeric_cols_to_round){ if(rcfg %in% names(display_data) && is.numeric(display_data[[rcfg]])) display_data[[rcfg]] <- round(display_data[[rcfg]], 2) }
    
    cols_order <- c(breakout_col_display_name, "Prod. Month")
    other_cols <- setdiff(names(display_data), cols_order)
    actual_cols_for_order <- intersect(c(cols_order, other_cols), names(display_data))
    if(breakout_col_display_name %in% names(display_data)){
      display_data <- display_data[, ..actual_cols_for_order]
    }
    
    
    DT::datatable(
      display_data,
      options = list(
        pageLength = 12,
        scrollX = TRUE,
        order=list(list(0, 'asc'), list(1,'desc')),
        rowCallback = dt_rowCallback_js
      ),
      rownames=F,
      caption = paste("Aggregated Monthly & Normalized Production by", breakout_col_display_name ,"(Map-Filtered Wells)")
    )
  })
  output$download_filtered_group_prod_data <- downloadHandler(
    filename = function() {
      breakout_col_r_name <- input$filtered_group_breakout_by
      breakout_col_display_name_fn <- names(which(c("OperatorName" = "Operator", "Formation" = "Formation", "FieldName" = "Field", "ProvinceState" = "Province_State", "FirstProdYear" = "First_Prod_Year") == breakout_col_r_name))
      if(length(breakout_col_display_name_fn)==0) breakout_col_display_name_fn <- breakout_col_r_name
      paste0("filtered_group_prod_by_", tolower(gsub("(/| )", "_", breakout_col_display_name_fn)), "_", Sys.Date(), ".csv")
    },
    content = function(file) {
      analysis_data <- filtered_group_cumulative_data()
      req(analysis_data, analysis_data$table_data)
      data_dl <- copy(analysis_data$table_data)
      
      if(!is.null(data_dl) && nrow(data_dl) > 0) {
        breakout_col_r_name <- input$filtered_group_breakout_by # Original R name
        breakout_col_dl_name_map <- c("OperatorName" = "Group_By_Operator",
                                      "Formation" = "Group_By_Formation",
                                      "FieldName" = "Group_By_Field",
                                      "ProvinceState" = "Group_By_Province_State",
                                      "FirstProdYear" = "Group_By_First_Prod_Year")
        breakout_col_dl_name <- breakout_col_dl_name_map[breakout_col_r_name]
        if(is.na(breakout_col_dl_name)) breakout_col_dl_name <- breakout_col_r_name
        
        
        old_dl_names <- c("PROD_DATE", breakout_col_r_name, "TotalMonthlyBOE", "AvgDailyBOE", "AvgDailyBOE_per_1000ft", "CumTotalMonthlyBOE", "TotalSumLateralLength")
        new_dl_names <- c("Prod_Month", breakout_col_dl_name, "Total_Monthly_BOE", "Avg_Daily_BOE", "Avg_Daily_BOE_per_1000ft", "Cumulative_BOE", "Sum_Producing_Lateral_Length_ft")
        
        # Ensure only existing columns are attempted to be renamed
        current_dl_names <- names(data_dl)
        valid_old_dl_names <- old_dl_names[old_dl_names %in% current_dl_names]
        valid_new_dl_names <- new_dl_names[match(valid_old_dl_names, old_dl_names)]
        
        if(length(valid_old_dl_names) > 0) {
          setnames(data_dl, old = valid_old_dl_names, new = valid_new_dl_names, skip_absent = TRUE)
        }
        fwrite(data_dl, file)
      } else { fwrite(data.table(Message="No data for download."), file) }
    }
  )
  
  # --- Type Curve Analysis (Arps) ---
  type_curve_analysis_data <- eventReactive(input$generate_type_curve, {
    req(wells_sf_global, input$product_type_filter_analysis)
    
    selected_analysis_products <- input$product_type_filter_analysis
    if (is.null(selected_analysis_products) || length(selected_analysis_products) == 0) {
      showNotification("Please select at least one product type for analysis (under 'Production Analysis' main tab).", type = "warning", duration=5)
      return(list(data=data.table(), fit=NULL, params_text_base="No product types selected for analysis.", eur_25yr = NA_real_, monthly_declines_for_output = data.table()))
    }
    
    showNotification("Generating Arps type curve data (peak normalized, daily rates)...", type = "message", duration = NULL, id="arpsMsg")
    filtered_wells_sf_for_arps <- reactive_vals$wells_to_display
    
    max_wells_for_type_curve <- 1000
    num_selected_wells <- if(!is.null(filtered_wells_sf_for_arps) && nrow(filtered_wells_sf_for_arps) > 0) {
      length(unique(na.omit(filtered_wells_sf_for_arps$GSL_UWI_Std)))
    } else { 0 }
    
    if (num_selected_wells > max_wells_for_type_curve) {
      showNotification(paste("Too many wells selected (", num_selected_wells, ") for type curve analysis. Please refine filters to select fewer than", max_wells_for_type_curve, "wells."), type = "warning", duration = 7, id="tooManyWellsArps")
      removeNotification("arpsMsg")
      return(list(data=data.table(), fit=NULL, params_text_base=paste("Too many wells selected (", num_selected_wells, "). Max allowed:", max_wells_for_type_curve), eur_25yr = NA_real_, monthly_declines_for_output = data.table()))
    }
    
    if (is.null(filtered_wells_sf_for_arps) || nrow(filtered_wells_sf_for_arps) == 0) { removeNotification("arpsMsg"); showNotification("No wells selected by map filters for type curve.", type = "warning"); return(list(data=data.table(), fit=NULL, params_text_base="No wells selected.", eur_25yr = NA_real_, monthly_declines_for_output = data.table())) }
    target_uwis <- unique(filtered_wells_sf_for_arps$GSL_UWI_Std)
    target_uwis <- target_uwis[!is.na(target_uwis) & target_uwis != ""]
    if (length(target_uwis) == 0) { removeNotification("arpsMsg"); showNotification("No valid GSL_UWI_Std for type curve.", type = "warning"); return(list(data=data.table(), fit=NULL, params_text_base="No valid UWIs.", eur_25yr = NA_real_, monthly_declines_for_output = data.table())) }
    if (is.null(con) || !dbIsValid(con)) { con <<- connect_to_db(); if (is.null(con) || !dbIsValid(con)) { removeNotification("arpsMsg"); return(list(data=data.table(), fit=NULL, params_text_base="DB connection failed.", eur_25yr = NA_real_, monthly_declines_for_output = data.table())) } }
    
    uwi_batches <- split(target_uwis, ceiling(seq_along(target_uwis)/300)); all_prod_list_arps <- list()
    for(batch_uwis in uwi_batches) {
      sql_arps <- glue::glue_sql( "SELECT GSL_UWI, YEAR, PRODUCT_TYPE, ACTIVITY_TYPE, JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME, MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, SEP_VOLUME, OCT_VOLUME, NOV_VOLUME, DEC_VOLUME FROM PDEN_VOL_BY_MONTH WHERE GSL_UWI IN ({uwis*}) AND ACTIVITY_TYPE = 'PRODUCTION' AND PRODUCT_TYPE IN ('OIL', 'CND', 'GAS')", uwis = batch_uwis, .con = con)
      batch_data <- tryCatch(as.data.table(dbGetQuery(con, sql_arps)), error = function(e) data.table()); if(nrow(batch_data)>0) all_prod_list_arps[[length(all_prod_list_arps)+1]] <- batch_data
    }
    if(length(all_prod_list_arps) == 0) { removeNotification("arpsMsg"); showNotification("No production data for type curve wells.", type="warning"); return(list(data=data.table(), fit=NULL, params_text_base="No production data.", eur_25yr = NA_real_, monthly_declines_for_output = data.table())) }
    full_prod_arps_raw <- rbindlist(all_prod_list_arps, use.names = TRUE, fill = TRUE)
    
    # Filter by selected product types for analysis (respecting "BOE" selection for Arps)
    effective_selected_analysis_products <- selected_analysis_products
    if ("BOE" %in% selected_analysis_products) {
      effective_selected_analysis_products <- unique(c(effective_selected_analysis_products, "OIL", "CND", "GAS"))
      effective_selected_analysis_products <- effective_selected_analysis_products[effective_selected_analysis_products != "BOE"]
    }
    
    if (!is.null(effective_selected_analysis_products) && length(effective_selected_analysis_products) > 0) {
      full_prod_arps_raw <- full_prod_arps_raw[toupper(PRODUCT_TYPE) %in% toupper(effective_selected_analysis_products)]
    }
    if(nrow(full_prod_arps_raw) == 0) {
      removeNotification("arpsMsg");
      showNotification("No production data for the selected product type(s) for Arps.", type = "warning");
      return(list(data=data.table(), fit=NULL, params_text_base="No production data for selected product(s).", eur_25yr = NA_real_, monthly_declines_for_output = data.table()))
    }
    
    if("GSL_UWI" %in% names(full_prod_arps_raw)) { full_prod_arps_raw[, GSL_UWI_Std_query := standardize_uwi(GSL_UWI)] }
    prod_arps_cleaned <- clean_df_colnames(full_prod_arps_raw, "Arps Type Curve PDEN")
    
    if ("GSL_UWI_STD_QUERY" %in% names(prod_arps_cleaned)) {
      setnames(prod_arps_cleaned, "GSL_UWI_STD_QUERY", "GSL_UWI_Std")
    } else if ("GSL_UWI" %in% names(prod_arps_cleaned) && !"GSL_UWI_Std" %in% names(prod_arps_cleaned)) {
      prod_arps_cleaned[, GSL_UWI_Std := standardize_uwi(GSL_UWI)]
    } else if (!"GSL_UWI_Std" %in% names(prod_arps_cleaned)) {
      warning("Arps: GSL_UWI_Std column not found after cleaning.")
      removeNotification("arpsMsg"); return(list(data=data.table(), fit=NULL, params_text_base="Cannot ID wells.", eur_25yr = NA_real_, monthly_declines_for_output = data.table()))
    }
    
    year_col <- "YEAR"; product_col <- "PRODUCT_TYPE";
    arps_product_choice <- input$arps_product_type # "Oil" or "Gas" (for Arps specific selection)
    
    # Determine which DB product types map to the Arps selection, considering the global analysis filter
    db_products_for_this_arps_run <- character(0)
    if (arps_product_choice == "Oil") { # This means "Oil/Condensate" for Arps
      if ("OIL" %in% effective_selected_analysis_products) db_products_for_this_arps_run <- c(db_products_for_this_arps_run, "OIL")
      if ("CND" %in% effective_selected_analysis_products) db_products_for_this_arps_run <- c(db_products_for_this_arps_run, "CND")
    } else if (arps_product_choice == "Gas") {
      if ("GAS" %in% effective_selected_analysis_products) db_products_for_this_arps_run <- c(db_products_for_this_arps_run, "GAS")
    }
    
    if (length(db_products_for_this_arps_run) == 0) {
      removeNotification("arpsMsg");
      showNotification(paste("No data for Arps. Arps product:", arps_product_choice, " is not among globally selected products:", paste(selected_analysis_products, collapse=", ")), type="warning", duration=7);
      return(list(data=data.table(), fit=NULL, params_text_base="No data for Arps product based on global filter.", eur_25yr = NA_real_, monthly_declines_for_output = data.table()))
    }
    
    month_cols <- paste0(toupper(month.abb), "_VOLUME")
    
    cols_to_check_arps <- c(year_col, product_col, "GSL_UWI_Std")
    if(!all(cols_to_check_arps %in% names(prod_arps_cleaned)) || !all(month_cols %in% names(prod_arps_cleaned))) {
      missing_cols_arps <- c(cols_to_check_arps[!cols_to_check_arps %in% names(prod_arps_cleaned)], month_cols[!month_cols %in% names(prod_arps_cleaned)])
      warning(paste("Arps: Missing required columns after cleaning:", paste(missing_cols_arps, collapse=", ")))
      removeNotification("arpsMsg"); return(list(data=data.table(), fit=NULL, params_text_base="Missing critical columns for Arps.", eur_25yr = NA_real_, monthly_declines_for_output = data.table()))
    }
    
    for(mc in month_cols) if(!is.numeric(prod_arps_cleaned[[mc]])) prod_arps_cleaned[, (mc) := as.numeric(get(mc))]
    prod_arps_cleaned[, (year_col) := as.numeric(get(year_col))]; prod_arps_cleaned[, (product_col) := as.character(get(product_col))]
    prod_arps_long <- melt(prod_arps_cleaned, id.vars = c("GSL_UWI_Std", year_col, product_col), measure.vars = month_cols, variable.name = "MONTH_VOL_COL", value.name = "MonthlyVolume", na.rm = FALSE)
    
    prod_arps_long <- prod_arps_long[toupper(get(product_col)) %in% toupper(db_products_for_this_arps_run)]
    if(nrow(prod_arps_long) == 0) { removeNotification("arpsMsg"); showNotification("No production data for selected Arps product type after global product filter.", type="warning"); return(list(data=data.table(), fit=NULL, params_text_base="No data for Arps product.", eur_25yr = NA_real_, monthly_declines_for_output = data.table())) }
    
    # Apply unit conversions
    prod_arps_long[toupper(get(product_col)) %in% c("OIL", "CND"), MonthlyVolume := MonthlyVolume * M3_TO_BBL]
    prod_arps_long[toupper(get(product_col)) == "GAS", MonthlyVolume := MonthlyVolume * E3M3_TO_MCF]
    
    # Aggregate volumes if "Oil/Condensate" is selected for Arps (summing OIL and CND for each well-month)
    if (arps_product_choice == "Oil" && length(intersect(c("OIL", "CND"), db_products_for_this_arps_run)) > 0) {
      prod_arps_long <- prod_arps_long[, .(MonthlyVolume = sum(MonthlyVolume, na.rm = TRUE)),
                                       by = .(GSL_UWI_Std, YEAR, MONTH_VOL_COL)]
      prod_arps_long[, (product_col) := "Oil_Cond_Combined"]
    }
    
    prod_arps_long <- prod_arps_long[!is.na(MonthlyVolume) & MonthlyVolume > 0]
    if(nrow(prod_arps_long) == 0) { removeNotification("arpsMsg"); showNotification("No positive production after unit conversion for Arps.", type="warning"); return(list(data=data.table(), fit=NULL, params_text_base="No positive production.", eur_25yr = NA_real_, monthly_declines_for_output = data.table())) }
    
    prod_arps_long[, Month_Num := match(toupper(substr(MONTH_VOL_COL, 1, 3)), toupper(month.abb))]
    prod_arps_long <- prod_arps_long[!is.na(get(year_col)) & !is.na(Month_Num)]
    prod_arps_long[, PROD_DATE := as.Date(paste(get(year_col), Month_Num, 1, sep="-"), format="%Y-%m-%d")]; prod_arps_long <- prod_arps_long[!is.na(PROD_DATE)]
    
    prod_arps_long[, DaysInMonth := lubridate::days_in_month(PROD_DATE)]
    prod_arps_long[, DailyRate := MonthlyVolume / DaysInMonth]
    prod_arps_long <- prod_arps_long[!is.na(DailyRate) & is.finite(DailyRate) & DailyRate > 0]
    
    if(nrow(prod_arps_long) == 0) { removeNotification("arpsMsg"); showNotification(paste("No", input$arps_product_type, "production for Arps after all filters."), type="warning"); return(list(data=data.table(), fit=NULL, params_text_base=paste("No", input$arps_product_type, "data."), eur_25yr = NA_real_, monthly_declines_for_output = data.table())) }
    
    if (!"GSL_UWI_Std" %in% names(prod_arps_long)) {
      warning("Arps: GSL_UWI_Std is missing before peak calculation.")
      removeNotification("arpsMsg"); return(list(data=data.table(), fit=NULL, params_text_base="GSL_UWI_Std missing for peak calc.", eur_25yr = NA_real_, monthly_declines_for_output = data.table()))
    }
    prod_arps_long[, PeakDailyRate := max(DailyRate, na.rm = TRUE), by = GSL_UWI_Std]
    peak_info <- prod_arps_long[DailyRate == PeakDailyRate, .(PeakProdDate = min(PROD_DATE)), by = GSL_UWI_Std]
    
    prod_arps_long_from_peak <- merge(prod_arps_long, peak_info, by = "GSL_UWI_Std", all.x = TRUE)
    prod_arps_long_from_peak <- prod_arps_long_from_peak[!is.na(PeakProdDate) & PROD_DATE >= PeakProdDate]
    
    if(nrow(prod_arps_long_from_peak) == 0) { removeNotification("arpsMsg"); showNotification("No production data at or after peak for Arps.", type="warning"); return(list(data=data.table(), fit=NULL, params_text_base="No data at/after peak.", eur_25yr = NA_real_, monthly_declines_for_output = data.table())) }
    
    prod_arps_long_from_peak[, MonthOnProdNormalized := round(as.numeric(PROD_DATE - PeakProdDate) / AVG_DAYS_PER_MONTH) + 1]
    
    type_curve_data <- prod_arps_long_from_peak[, .(
      AvgDailyRate = mean(DailyRate, na.rm = TRUE),
      WellCount = uniqueN(GSL_UWI_Std)
    ), by = MonthOnProdNormalized][order(MonthOnProdNormalized)]
    setnames(type_curve_data, "MonthOnProdNormalized", "MonthOnProd")
    
    type_curve_data <- type_curve_data[WellCount >= 3]
    if(nrow(type_curve_data) < 3) { removeNotification("arpsMsg"); return(list(data=type_curve_data, fit=NULL, params_text_base="Not enough data points for fitting post-peak.", eur_25yr = NA_real_, monthly_declines_for_output = data.table())) }
    
    initial_Qi <- type_curve_data$AvgDailyRate[1]
    if (nrow(type_curve_data) >= 2 && initial_Qi > 0 && is.finite(initial_Qi)) { # Added is.finite
      initial_Di_monthly <- abs( (type_curve_data$AvgDailyRate[2] - initial_Qi) / initial_Qi )
      if(is.na(initial_Di_monthly) || !is.finite(initial_Di_monthly) || initial_Di_monthly == 0) initial_Di_monthly <- 0.05
    } else { initial_Di_monthly <- 0.05 }
    if(is.na(initial_Qi) || !is.finite(initial_Qi) || initial_Qi <=0) initial_Qi <- max(type_curve_data$AvgDailyRate[is.finite(type_curve_data$AvgDailyRate)], na.rm=T); if(is.na(initial_Qi) || !is.finite(initial_Qi) || initial_Qi <=0) initial_Qi <- 100 # Ensure finite max
    
    arps_params_text <- "Fit failed or not attempted."; fit_model <- NULL; eur_25yr <- NA_real_;
    monthly_declines_for_output <- data.table()
    
    tryCatch({
      if(input$arps_model_type == "hyperbolic") { fit_model <- nls(AvgDailyRate ~ Qi / (1 + b * Di * MonthOnProd)^(1/b), data = type_curve_data, start = list(Qi = initial_Qi, Di = initial_Di_monthly, b = 1.0), lower = list(Qi=1e-3, Di=1e-6, b=1e-6), upper = list(Qi=Inf, Di=1, b=2.0), algorithm = "port", control = nls.control(maxiter=200, warnOnly = TRUE, minFactor=1/2048))
      } else if (input$arps_model_type == "exponential") { fit_model <- nls(AvgDailyRate ~ Qi * exp(-Di * MonthOnProd), data = type_curve_data, start = list(Qi = initial_Qi, Di = initial_Di_monthly), lower = list(Qi=1e-3, Di=1e-6), algorithm = "port", control = nls.control(maxiter=100, warnOnly = TRUE))
      } else { fit_model <- nls(AvgDailyRate ~ Qi / (1 + Di * MonthOnProd), data = type_curve_data, start = list(Qi = initial_Qi, Di = initial_Di_monthly), lower = list(Qi=1e-3, Di=1e-6), algorithm = "port", control = nls.control(maxiter=100, warnOnly = TRUE)) }
      
      if(!is.null(fit_model)){
        # EUR Calculation with Dmin
        forecast_months <- 1:(25*12) # 25 years
        predicted_rates_dt <- data.table(MonthOnProd = forecast_months)
        predicted_rates_dt[, Rate_Arps := predict(fit_model, newdata = .SD)]
        predicted_rates_dt[Rate_Arps < 0 | is.na(Rate_Arps), Rate_Arps := 0]
        
        # Determine Dmin (annual effective)
        d_min_annual_eff <- if (input$arps_product_type == "Oil") 0.10 else 0.08 # 10% for Oil/Cnd, 8% for Gas
        # Convert annual effective Dmin to monthly effective Dmin
        d_min_monthly_eff <- 1 - (1 - d_min_annual_eff)^(1/12)
        
        # Apply Dmin from Year 11 (Month 121) onwards
        rate_at_month_120 <- predicted_rates_dt[MonthOnProd == 120, Rate_Arps]
        
        predicted_rates_dt[, FinalRate := Rate_Arps] # Initialize with Arps rate
        
        if (nrow(predicted_rates_dt[MonthOnProd > 120]) > 0 && rate_at_month_120 > 0) { # Check if rate_at_month_120 is positive
          # For months > 120, apply Dmin based on the rate of the PREVIOUS month using Dmin
          # Initialize rate for month 121 using Dmin from rate_at_month_120
          predicted_rates_dt[MonthOnProd == 121, FinalRate := rate_at_month_120 * (1 - d_min_monthly_eff)]
          
          for (m in 122:max(forecast_months)) {
            rate_prev_month_dmin <- predicted_rates_dt[MonthOnProd == (m - 1), FinalRate]
            predicted_rates_dt[MonthOnProd == m, FinalRate := rate_prev_month_dmin * (1 - d_min_monthly_eff)]
          }
        }
        predicted_rates_dt[FinalRate < 0 | is.na(FinalRate), FinalRate := 0] # Ensure no negative rates
        
        predicted_monthly_volumes_eur <- predicted_rates_dt$FinalRate * AVG_DAYS_PER_MONTH
        eur_25yr <- sum(predicted_monthly_volumes_eur, na.rm = TRUE)
        
        # For decline output, use the Arps-fitted curve for the first 10 years
        max_decline_month_output <- 120
        df_rates_for_decline_output <- data.table(MonthOnProd = 0:max_decline_month_output)
        df_rates_for_decline_output$Rate <- predict(fit_model, newdata = df_rates_for_decline_output)
        df_rates_for_decline_output[Rate < 0 | is.na(Rate), Rate := 0]
        df_rates_for_decline_output[, Rate_Prev_Month := shift(Rate, n=1, fill=NA, type="lag")]
        df_rates_for_decline_output <- df_rates_for_decline_output[MonthOnProd > 0]
        df_rates_for_decline_output[, MonthlyEffectiveDecline := ifelse(Rate_Prev_Month > 1e-6, (Rate_Prev_Month - Rate) / Rate_Prev_Month, NA_real_)] # Avoid division by zero/tiny
        monthly_declines_for_output <- df_rates_for_decline_output[, .(MonthOnProd, MonthlyEffectiveDecline)]
        
      }
    }, error = function(e) { arps_params_text <<- paste("Arps fitting error:", e$message); fit_model <<- NULL; eur_25yr <<- NA_real_; monthly_declines_for_output <<- data.table() })
    
    removeNotification("arpsMsg"); showNotification("Type curve data processed.", type="message")
    return(list(data = type_curve_data, fit = fit_model, params_text_base = arps_params_text, eur_25yr = eur_25yr, monthly_declines_for_output = monthly_declines_for_output))
  })
  
  output$arps_type_curve_plot <- renderPlot({
    analysis_results <- type_curve_analysis_data()
    req(analysis_results, analysis_results$data)
    plot_data <- analysis_results$data
    fit <- analysis_results$fit
    if(nrow(plot_data) == 0) { return(ggplot() + labs(title = "No data for Arps type curve.", x="Months Since Peak Production", y="Average Daily Rate") + theme_void()) }
    
    y_axis_label <- paste("Average Daily Rate", ifelse(input$arps_product_type=="Oil", "(BBL/day)", "(MCF/day)"))
    
    g <- ggplot(plot_data, aes(x = MonthOnProd, y = AvgDailyRate)) +
      geom_point(aes(size=WellCount), alpha=0.7, color=custom_palette[1]) +
      geom_line(color=custom_palette[1], alpha=0.5) +
      scale_y_continuous(labels = scales::comma) +
      scale_x_continuous(breaks = scales::pretty_breaks(n=10)) +
      labs(title = paste("Arps Type Curve (Peak Normalized - Daily Rates) -", input$arps_product_type, "(", input$arps_model_type, "model)"),
           x = "Months Since Peak Production", y = y_axis_label,
           size = "Well Count",
           caption = "Points: Avg Actual Daily Rate. Dashed Red Line: Fitted Arps Curve.") +
      theme_minimal(base_size = 12) + theme(legend.position = "top")
    if (!is.null(fit)) {
      max_month_plot <- max(plot_data$MonthOnProd, na.rm = TRUE)
      predict_months_plot <- seq(1, max_month_plot, length.out = 200)
      if(is.finite(max_month_plot) && max_month_plot > 0){
        predict_df_plot <- data.table(MonthOnProd = predict_months_plot)
        tryCatch({
          predict_df_plot$PredictedRate <- predict(fit, newdata = predict_df_plot)
          predict_df_plot[PredictedRate < 0, PredictedRate := 0]
          g <- g + geom_line(data = predict_df_plot, aes(x=MonthOnProd, y = PredictedRate), color = custom_palette[2], linewidth = 1.1, linetype="dashed")
        }, error = function(e) { message(paste("Error during predict for Arps plot:", e$message)) })
      }
    }
    return(g)
  })
  
  output$arps_parameters_output <- renderText({
    analysis_results <- type_curve_analysis_data()
    req(analysis_results)
    
    params_text_list <- list()
    fit_model <- analysis_results$fit
    type_curve_data_for_calc <- analysis_results$data
    
    if (input$arps_product_type == "Oil") {
      params_text_list[["Product Type Note"]] <- "Oil/Condensate selection includes 'OIL' and 'CND' product types from database (if selected in global product filter)."
    }
    
    if (!is.null(fit_model)) {
      params <- coef(fit_model)
      params_text_list[["Fitted Model"]] <- paste("Model:", input$arps_model_type)
      params_text_list[["Qi (Initial Daily Rate of Fitted Curve)"]] <- paste(signif(params[["Qi"]], 4), ifelse(input$arps_product_type=="Oil", "BBL/day", "MCF/day"))
      
      if (nrow(type_curve_data_for_calc[MonthOnProd == 1]) > 0) {
        actual_ip30_val <- type_curve_data_for_calc[MonthOnProd == 1, AvgDailyRate]
        params_text_list[["Avg. Daily Rate Month 1 (IP30 Approx.)"]] <- paste(round(actual_ip30_val, 1), ifelse(input$arps_product_type=="Oil", "BBL/day", "MCF/day"))
      } else {
        params_text_list[["Avg. Daily Rate Month 1 (IP30 Approx.)"]] <- "N/A (No data at Month 1)"
      }
      
      params_text_list[["Di (Nominal Monthly Decline of Fitted Curve)"]] <- signif(params[["Di"]], 4)
      
      b_val_text <- if ("b" %in% names(params)) {
        signif(params[["b"]], 4)
      } else if (input$arps_model_type == "exponential"){
        "0 (Exponential)"
      } else if (input$arps_model_type == "harmonic") {
        "1 (Harmonic)"
      } else { "N/A" }
      params_text_list[["b (Hyperbolic Exponent of Fitted Curve)"]] <- b_val_text
      
      if ("b" %in% names(params) && params[["b"]] < 0.001 && params[["b"]] > -0.001 && input$arps_model_type == "hyperbolic") {
        params_text_list[["Note on 'b'"]] <- "Fitted 'b' is very small; curve behaves like an Exponential decline. This will result in constant effective decline percentages for periods of the same length."
      }
      
      eur_val <- analysis_results$eur_25yr
      eur_unit <- ifelse(input$arps_product_type=="Oil", "MBBL", "MMCF")
      eur_display_val <- if(!is.na(eur_val)) eur_val / 1000 else NA_real_
      
      if(!is.na(eur_display_val)) {
        params_text_list[["25-Year EUR (Adjusted for Terminal Decline)"]] <- paste(formatC(eur_display_val, format="f", big.mark=",", digits=1), eur_unit)
      } else {
        params_text_list[["25-Year EUR (Adjusted for Terminal Decline)"]] <- "N/A"
      }
      
    } else {
      params_text_list[["Fitted Model"]] <- analysis_results$params_text_base
    }
    
    # --- Decline Rate Percentages based on Excel Logic ---
    decline_output_formatted_lines <- list()
    decline_output_formatted_lines[[1]] <- "Decline Period\tRange Percentage"
    
    if (!is.null(fit_model) && nrow(analysis_results$monthly_declines_for_output) > 0) {
      monthly_eff_declines_dt <- analysis_results$monthly_declines_for_output
      
      get_d_eff <- function(month_num) {
        val <- monthly_eff_declines_dt[MonthOnProd == month_num, MonthlyEffectiveDecline]
        if (length(val) == 0 || is.na(val)) return(NA_real_)
        return(val)
      }
      
      get_rate_at_end_of_month <- function(month_num_on_prod) {
        if (month_num_on_prod == 0) return(coef(fit_model)[["Qi"]])
        val <- predict(fit_model, newdata = data.table(MonthOnProd = month_num_on_prod))
        if (length(val) == 0 || is.na(val) || val < 0) return(0)
        return(val)
      }
      
      decline_periods_monthly_geom <- list(
        "Month 2"   = c(2),
        "Month 3-4"   = c(3, 4),
        "Month 5-6"   = c(5, 6),
        "Month 7-12"  = 7:12,
        "Month 13-18" = 13:18,
        "Month 19-24" = 19:24,
        "Month 25-36" = 25:36
      )
      
      for(period_name in names(decline_periods_monthly_geom)){
        months_in_period <- decline_periods_monthly_geom[[period_name]]
        declines_for_geom_mean <- sapply(months_in_period, get_d_eff)
        valid_declines <- declines_for_geom_mean[!is.na(declines_for_geom_mean) & declines_for_geom_mean > 0]
        
        period_decline_val <- if (length(valid_declines) > 0) {
          if (length(months_in_period) == 1 && period_name == "Month 2") {
            valid_declines[1]
          } else if (length(valid_declines) == length(months_in_period)) {
            geometric_mean(valid_declines, na.rm = TRUE)
          } else {
            NA_real_
          }
        } else {
          NA_real_
        }
        decline_output_formatted_lines[[length(decline_output_formatted_lines) + 1]] <-
          paste0(period_name, "\t", ifelse(is.na(period_decline_val), "N/A", paste0(format(round(period_decline_val * 100, 2), nsmall = 2), "%")))
      }
      decline_output_formatted_lines[[length(decline_output_formatted_lines) + 1]] <- "" # Cell break for Excel
      
      yearly_decline_periods <- list(
        "Year 4" = c(36, 48),
        "Year 5" = c(48, 60)
      )
      for(period_name in names(yearly_decline_periods)){
        m_start <- yearly_decline_periods[[period_name]][1]
        m_end   <- yearly_decline_periods[[period_name]][2]
        
        rate_at_period_start <- get_rate_at_end_of_month(m_start)
        rate_at_period_end   <- get_rate_at_end_of_month(m_end)
        
        period_decline_val <- if (rate_at_period_start > 1e-6) {
          (rate_at_period_start - rate_at_period_end) / rate_at_period_start
        } else { NA_real_ }
        decline_output_formatted_lines[[length(decline_output_formatted_lines) + 1]] <-
          paste0(period_name, "\t", ifelse(is.na(period_decline_val), "N/A", paste0(format(round(period_decline_val * 100, 2), nsmall = 2), "%")))
      }
      
      annual_declines_yr6_10 <- numeric()
      for (year_num in 6:10) {
        m_start_of_year <- (year_num - 1) * 12
        m_end_of_year   <- year_num * 12
        
        rate_at_year_start <- get_rate_at_end_of_month(m_start_of_year)
        rate_at_year_end   <- get_rate_at_end_of_month(m_end_of_year)
        
        annual_decline <- if (rate_at_year_start > 1e-6) {
          (rate_at_year_start - rate_at_year_end) / rate_at_year_start
        } else { NA_real_ }
        annual_declines_yr6_10 <- c(annual_declines_yr6_10, annual_decline)
      }
      valid_annual_declines_yr6_10 <- annual_declines_yr6_10[!is.na(annual_declines_yr6_10) & annual_declines_yr6_10 > 0]
      geom_mean_yr6_10 <- if (length(valid_annual_declines_yr6_10) > 0) {
        geometric_mean(valid_annual_declines_yr6_10, na.rm = TRUE)
      } else { NA_real_ }
      decline_output_formatted_lines[[length(decline_output_formatted_lines) + 1]] <-
        paste0("Year 6-10", "\t", ifelse(is.na(geom_mean_yr6_10), "N/A", paste0(format(round(geom_mean_yr6_10 * 100, 2), nsmall = 2), "%")))
      
      # Year 11+ text based on user request
      year_11_plus_decline_text <- if (input$arps_product_type == "Oil") {
        "Terminal Decline (Dmin): 10.00% (Annual Eff.)"
      } else { # Gas
        "Terminal Decline (Dmin): 8.00% (Annual Eff.)"
      }
      decline_output_formatted_lines[[length(decline_output_formatted_lines) + 1]] <- paste0("Year 11+", "\t", year_11_plus_decline_text)
      
    } else {
      decline_output_formatted_lines[[length(decline_output_formatted_lines) + 1]] <- "Decline Rates: Not calculated (fitting or monthly declines failed)."
    }
    
    main_params_output <- paste(names(params_text_list), params_text_list, sep = ": ", collapse = "\n")
    main_params_output <- gsub("Fitted Model: Model:", "Fitted Model:", main_params_output, fixed=TRUE)
    
    final_output_string <- paste0(
      main_params_output,
      "\n\n--- Decline Rate Percentages ---\n",
      paste(decline_output_formatted_lines, collapse = "\n")
    )
    
    return(final_output_string)
  })
  
  output$arps_data_table <- DT::renderDataTable({
    analysis_results <- type_curve_analysis_data()
    req(analysis_results, analysis_results$data)
    dt_data <- copy(analysis_results$data)
    if("AvgDailyRate" %in% names(dt_data)) setnames(dt_data, "AvgDailyRate", "Avg.Daily.Rate")
    
    if(!is.null(analysis_results$fit)){
      tryCatch({
        dt_data$PredictedDailyRate <- predict(analysis_results$fit, newdata=dt_data)
      }, error = function(e) {
        message(paste("Error during predict for Arps table:", e$message))
        dt_data$PredictedDailyRate <- NA_real_
      })
    }
    numeric_cols <- names(dt_data)[sapply(dt_data, is.numeric)]
    for(col in numeric_cols) {
      if (col %in% c("Avg.Daily.Rate", "PredictedDailyRate")) {
        dt_data[, (col) := round(get(col), 1)]
      } else if (col != "WellCount") {
        if (col %in% names(dt_data)) {
          dt_data[, (col) := signif(get(col), 4)]
        }
      }
    }
    if("MonthOnProd" %in% names(dt_data)) setnames(dt_data, "MonthOnProd", "MonthsSincePeak")
    
    display_cols <- c("MonthsSincePeak", "Avg.Daily.Rate", "WellCount")
    if("PredictedDailyRate" %in% names(dt_data)) display_cols <- c(display_cols, "PredictedDailyRate")
    
    display_cols_exist <- display_cols[display_cols %in% names(dt_data)]
    
    DT::datatable(
      dt_data[, ..display_cols_exist],
      options = list(
        pageLength = 10,
        scrollX = TRUE,
        rowCallback = dt_rowCallback_js
      ),
      rownames = FALSE,
      caption = paste("Aggregated Data for Arps Type Curve (Daily Rates -", ifelse(input$arps_product_type=="Oil","BBL/day","MCF/day"), ")")
    )
  })
  
} # End of server function

# --- 6. Run the Application ---
shinyApp(ui = ui, server = server)


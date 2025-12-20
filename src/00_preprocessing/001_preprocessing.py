import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import from_wkb
import psutil 

# ============= 0. BASE CONFIGURATION ===================

BASE_DIR = Path.cwd()
print("Base dir:", BASE_DIR)
PROJECT_ROOT = BASE_DIR.parents[1]

WORLD_BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "boundaries"
    / "world"
    / "world_boundaries_overture_iso_a3.parquet"
)

EXPORT_DAILY_PREFIX = PROJECT_ROOT / "results" / "00_preprocessing" / "daily" / "user_daily_activity"

# DuckDB resource config (for CLI)
total_ram_gb = psutil.virtual_memory().total / 1024**3
duckdb_ram_gb = max(4, int(total_ram_gb * 0.8))  # use ~60% of RAM, at least 4GB
num_threads = max(1, (os.cpu_count() or 4) - 2)   # leave 2 cores free

MAX_MEMORY_SQL = f"""
SET MAX_MEMORY = '{duckdb_ram_gb}GB';
SET THREADS = {num_threads};
PRAGMA enable_progress_bar = true;
"""

# Defining Parquet Path from Ohsome Planet ----IMPORTANT if there are changes--------
#-----------------  IF DATA SOURCE CHANGES, UPDATE THIS PATH  -----------------
PATH_OHSOME_PLANET = "s3://heigit-ohsome-planet/data/global/2025-07-29/*/*.parquet"

# ============= 0.1 LOAD WORLD BOUNDARIES ===============

df = pd.read_parquet(WORLD_BOUNDARY_FILE)
df = df[df["iso"].str.len() > 1]
df["geometry"] = df["geometry"].apply(from_wkb)
WORLD_GDF = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

print("Export prefix for daily user activity:", EXPORT_DAILY_PREFIX)


# ============= 1. HELPER FUNCTIONS =====================

def build_iso_condition(iso_codes: str) -> str:
    """
    Builds a SQL condition from a list of ISO codes. Uses special handling for 'STATELESS' and 'SPECIAL_REGIONS'.
    """
    if iso_codes == 'STATELESS':
        condition = "len(country_iso_a3) = 0"
        return condition
    elif iso_codes == 'SPECIAL_REGIONS':
        condition = """(array_contains(countries, 'CP') OR
                array_contains(countries, 'XA') OR
                array_contains(countries, 'XB') OR
                array_contains(countries, 'XC') OR
                array_contains(countries, 'XD') OR
                array_contains(countries, 'XE') OR
                array_contains(countries, 'XG') OR
                array_contains(countries, 'XH') OR
                array_contains(countries, 'XI') OR
                array_contains(countries, 'XJ') OR
                array_contains(countries, 'XK') OR
                array_contains(countries, 'XL') OR
                array_contains(countries, 'XM') OR
                array_contains(countries, 'XN') OR
                array_contains(countries, 'XO') OR
                array_contains(countries, 'XP') OR
                array_contains(countries, 'XQ') OR
                array_contains(countries, 'XR') OR
                array_contains(countries, 'XS') OR
                array_contains(countries, 'XT') OR
                array_contains(countries, 'XU') OR
                array_contains(countries, 'XW') OR
                array_contains(countries, 'XX') OR
                array_contains(countries, 'XY') OR
                array_contains(countries, 'XZ'))"""
        return condition
    else:
        condition = f"array_contains(country_iso_a3, '{iso_codes}')"
        return condition

def get_country_bbox(iso_code: str, gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Return bounding box [xmin, ymin, xmax, ymax] for the ISO3 code.
    'STATELESS' and 'SPECIAL_REGIONS' get global bounds.
    """
    iso_code = iso_code.upper()

    if iso_code in ("STATELESS", "SPECIAL_REGIONS"):
        return np.array([-180.0, -90.0, 180.0, 90.0])

    subset = gdf[gdf["iso"].isin([iso_code])]
    if subset.empty:
        raise ValueError(f"No countries found for ISO A3 code: {iso_code}")
    return subset.total_bounds  # [xmin, ymin, xmax, ymax]


def build_secret_sql(access_key: str, secret_key: str) -> str:
    """
    Build the S3 SECRET definition for DuckDB.
    """
    return f"""
DROP SECRET IF EXISTS secret;
CREATE SECRET secret (
    TYPE S3,
    KEY_ID '{access_key}',
    SECRET '{secret_key}',
    REGION 'eu-central-1',
    endpoint 'hot.storage.heigit.org',
    use_ssl true,
    url_style 'path'
);
"""


# ============= 2. SQL BUILDERS =========================
def build_enhancement_query(iso_codes_str: str, condition: str, bbox: np.ndarray) -> str:
    """
    Build the SQL that creates osm_data_{iso_code} with all enhancements.
    Paste your existing enhancement_query body inside.
    """
    xmin, ymin, xmax, ymax = bbox

    return f"""
    CREATE OR REPLACE TABLE osm_data_{iso_codes_str} AS
	WITH base AS (
  SELECT
    countries AS country_iso_a3,
    osm_id,
    user.id AS user_id,
    valid_from,
    valid_to,
    tags,
    tags_before,
    contrib_type,
    changeset,
    bbox,
    osm_type
  FROM read_parquet('{PATH_OHSOME_PLANET}')
  WHERE
    {condition} AND
    bbox.xmin >= { xmin } AND 
    bbox.xmax <= { xmax } AND 
    bbox.ymin >= { ymin } AND
    bbox.ymax <= { ymax } AND
    valid_from >= TIMESTAMP '2007-01-01T00:00:00' AND
    valid_from < TIMESTAMP '2025-07-29T00:00:00'
),
diffs AS (
  SELECT
    *,
    map_keys(tags) AS tag_keys,
    map_keys(tags_before) AS tag_before_keys,

    LIST_FILTER(map_keys(tags), k -> NOT CONTAINS(map_keys(tags_before), k)) AS new_keys,
    LIST_FILTER(map_keys(tags_before), k -> NOT CONTAINS(map_keys(tags), k)) AS removed_keys,

    LIST_TRANSFORM(
      LIST_FILTER(map_entries(tags), kv -> NOT map_contains(tags_before, kv.key)),
      kv -> CASE
        WHEN kv.key = 'building' AND kv.value != 'no' THEN 'building'
        WHEN kv.key = 'highway' THEN 'road'
        WHEN kv.key = 'amenity' THEN 'amenity'
        WHEN kv.key = 'water' AND kv.value IN ('pond','lake','reservoir','basin','oxbow','lagoon','fishpond') THEN 'body_of_water'
        WHEN kv.key = 'shop' THEN 'shop'
        WHEN kv.key = 'amenity' AND kv.value IN ('kindergarten','school','college','university') THEN 'educational_institution'
        WHEN kv.key = 'building' AND kv.value IN ('kindergarten','school','college','university') THEN 'educational_institution'
        WHEN kv.key = 'amenity' AND kv.value IN ('atm','bank','money_transfer','bureau_de_change','mobile_money_agent','payment_terminal') THEN 'financial_service'
        WHEN kv.key = 'healthcare' THEN 'healthcare_facility'
        WHEN kv.key = 'amenity' AND kv.value IN ('doctors','dentist','clinic','hospital','pharmacy') THEN 'healthcare_facility'
        WHEN kv.key = 'landuse' THEN 'land_use'
        WHEN kv.key = 'natural' AND kv.value IN ('bare_rock','beach','dune','fell','glacier','grassland','heath','landslide','mud','rock','sand','scree','scrub','shingle','water','wetland','wood') THEN 'land_use'
        WHEN kv.key = 'waterway' AND kv.value IN ('boatyard','dam','dock','riverbank') THEN 'land_use'
        WHEN kv.key = 'place' AND kv.value IN ('country','state','region','province','district','county','municipality','city','borough','suburb','quarter','neighbourhood','town','village','hamlet','isolated_dwelling') THEN 'place'
        WHEN kv.key IN ('amenity','shop','craft','office','leisure','aeroway') THEN 'point_of_interest'
        WHEN kv.key = 'social_facility' THEN 'social_facility'
        WHEN kv.key = 'amenity' AND kv.value IN ('shelter','social_facility','refugee_site') THEN 'social_facility'
        WHEN kv.key = 'amenity' AND kv.value IN ('toilets','shower','drinking_water','water_point') THEN 'wash_facility'
        WHEN kv.key = 'man_made' AND kv.value IN ('water_tap','borehole','water_works','pumping_station','pump','wastewater_plant','storage_tank','water_well','water_tower','reservoir_covered','water_tank') THEN 'wash_facility'
        WHEN kv.key = 'waterway' AND kv.value IN ('river','canal','stream','brook','drain','ditch') THEN 'waterway'
        ELSE 'other'
      END
    ) AS new_feature_categories,

    LIST_TRANSFORM(
      LIST_FILTER(map_entries(tags_before), kv -> NOT map_contains(tags, kv.key)),
      kv -> CASE
        WHEN kv.key = 'building' AND kv.value != 'no' THEN 'building'
        ELSE 'other'
      END
    ) AS removed_feature_categories
  FROM base
),
final AS (
  SELECT
    country_iso_a3,
    osm_id,
    user_id,
    valid_from,
    valid_to,
    contrib_type,
    tag_keys,
    new_keys,
    removed_keys,
    new_feature_categories,
    removed_feature_categories,
    length(new_keys) AS length_new_keys,
    length(removed_keys) AS length_removed_keys,
    tags,
    tags_before,
    changeset.id AS changeset_id,
    changeset.editor,
    len(changeset.tags['comment']) AS length_comment,
    changeset.closed_at - changeset.created_at AS changeset_edit_duration,
    CASE 
      WHEN changeset.editor ILIKE '%josm%' THEN 'JOSM'
      WHEN changeset.editor ILIKE 'iD%' THEN 'iD'
      WHEN changeset.editor ILIKE '%streetcomplete%' THEN 'Streetcomplete'
      ELSE 'other'
    END AS editor_category,
    LEAST(valid_to, TIMESTAMP '2025-07-29 23:59:59') - valid_from AS validity_duration,
    osm_type,

    -- Feature flags
    tags['building'] IS NOT NULL AND tags['building'] != 'no' AS building,
    tags['highway'] IS NOT NULL AS road,
    tags['amenity'] IS NOT NULL AS amenity,
    tags['water'] IN ('pond','lake','reservoir','basin','oxbow','lagoon','fishpond') AS body_of_water,
    tags['shop'] IS NOT NULL AS shop,
    tags['amenity'] IN ('kindergarten','school','college','university') 
      OR tags['building'] IN ('kindergarten','school','college','university') AS educational_institution,
    tags['amenity'] IN ('atm','bank','money_transfer','bureau_de_change','mobile_money_agent','payment_terminal') AS financial_service,
    tags['healthcare'] IS NOT NULL 
      OR tags['amenity'] IN ('doctors','dentist','clinic','hospital','pharmacy') AS healthcare_facility,
    tags['landuse'] IS NOT NULL
      OR tags['natural'] IN ('bare_rock','beach','dune','fell','glacier','grassland','heath','landslide','mud','rock','sand','scree','scrub','shingle','water','wetland','wood') 
      OR tags['waterway'] IN ('boatyard','dam','dock','riverbank') AS land_use,
    tags['place'] IN ('country','state','region','province','district','county','municipality','city','borough','suburb','quarter','neighbourhood','town','village','hamlet','isolated_dwelling') AS place,
    (
      tags['amenity'] IS NOT NULL OR
      tags['shop'] IS NOT NULL OR
      tags['craft'] IS NOT NULL OR
      tags['office'] IS NOT NULL OR
      tags['leisure'] IS NOT NULL OR
      tags['aeroway'] IS NOT NULL
    ) AS point_of_interest,
    tags['social_facility'] IS NOT NULL 
      OR tags['amenity'] IN ('shelter','social_facility','refugee_site') AS social_facility,
    tags['amenity'] IN ('toilets','shower','drinking_water','water_point') 
      OR tags['man_made'] IN ('water_tap','borehole','water_works','pumping_station','pump','wastewater_plant','storage_tank','water_well','water_tower','reservoir_covered','water_tank') AS wash_facility,
    tags['waterway'] IN ('river','canal','stream','brook','drain','ditch') AS waterway

  FROM diffs
)
SELECT * FROM final;
"""

def build_daily_user_query(iso_codes_str: str) -> str:
    """
    Build the SQL that aggregates daily_per_user_{iso_code}.
    Paste your existing daily_user_query body inside.
    """
    return f"""
    CREATE or REPLACE table daily_per_user_{iso_codes_str} AS
  WITH daily_edits AS (
    SELECT
      user_id,
      valid_from::DATE AS edit_date,
      building,
      road,
      amenity,
      body_of_water,
      shop,
      educational_institution,
      financial_service,
      healthcare_facility,
      land_use,
      place,
      point_of_interest,
      social_facility,
      wash_facility,
      waterway,
      '{iso_codes_str}' as country,
      COUNT(*) AS contrib_count,
      contrib_type,
      osm_type,
      validity_duration
    FROM osm_data_{iso_codes_str}
    GROUP BY user_id, edit_date,osm_type, building, road, amenity, body_of_water, shop, educational_institution, financial_service, healthcare_facility, land_use, place, point_of_interest, social_facility, wash_facility, waterway, contrib_type, validity_duration
  ),
  summed AS (
      SELECT 
          user_id,
          edit_date,
          country,
          SUM(EXTRACT(EPOCH FROM validity_duration)) / 86400.0 AS total_days_valid,
          SUM(CASE WHEN osm_type='node'     THEN contrib_count ELSE 0 END) AS count_node_per_day,
          SUM(CASE WHEN osm_type='way'      THEN contrib_count ELSE 0 END) AS count_way_per_day,
          SUM(CASE WHEN osm_type='relation' THEN contrib_count ELSE 0 END) AS count_relation_per_day,
          
          -- Overall activity
          SUM(contrib_count) AS edit_count,

          -- contrib_types for all features
          SUM(CASE WHEN contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS count_created_per_day,
          SUM(CASE WHEN contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS count_deleted_per_day,
          SUM(CASE WHEN contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS count_tag_only_per_day,
          SUM(CASE WHEN contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS count_geom_only_per_day,
          SUM(CASE WHEN contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS count_tag_and_geom_per_day,
          SUM(CASE WHEN contrib_type = '' THEN contrib_count ELSE 0 END) AS count_none_per_day,


          -- Overall sums per feature
          SUM(CASE WHEN building THEN contrib_count ELSE 0 END) AS count_building_per_day,
          SUM(CASE WHEN road THEN contrib_count ELSE 0 END) AS count_road_per_day,
          SUM(CASE WHEN amenity THEN contrib_count ELSE 0 END) AS count_amenity_per_day,
          SUM(CASE WHEN body_of_water THEN contrib_count ELSE 0 END) AS count_body_of_water_per_day,
          SUM(CASE WHEN shop THEN contrib_count ELSE 0 END) AS count_shop_per_day,
          SUM(CASE WHEN educational_institution THEN contrib_count ELSE 0 END) AS count_educational_institution_per_day,
          SUM(CASE WHEN financial_service THEN contrib_count ELSE 0 END) AS count_financial_service_per_day,
          SUM(CASE WHEN healthcare_facility THEN contrib_count ELSE 0 END) AS count_healthcare_facility_per_day,
          SUM(CASE WHEN land_use THEN contrib_count ELSE 0 END) AS count_land_use_per_day,
          SUM(CASE WHEN place THEN contrib_count ELSE 0 END) AS count_place_per_day,
          SUM(CASE WHEN point_of_interest THEN contrib_count ELSE 0 END) AS count_point_of_interest_per_day,
          SUM(CASE WHEN social_facility THEN contrib_count ELSE 0 END) AS count_social_facility_per_day,
          SUM(CASE WHEN wash_facility THEN contrib_count ELSE 0 END) AS count_wash_facility_per_day,
          SUM(CASE WHEN waterway THEN contrib_count ELSE 0 END) AS count_waterway_per_day,

          -- Per feature and contrib_type
          -- BUILDING
          CAST(SUM(CASE WHEN building AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS building_created_per_day,
          CAST(SUM(CASE WHEN building AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS building_deleted_per_day,
          CAST(SUM(CASE WHEN building AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS building_tag_only_per_day,
          CAST(SUM(CASE WHEN building AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS building_geometry_only_per_day,
          CAST(SUM(CASE WHEN building AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS building_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN building AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS building_none_per_day,
          -- ROAD
          CAST(SUM(CASE WHEN road AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS road_created_per_day,
          CAST(SUM(CASE WHEN road AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS road_deleted_per_day,
          CAST(SUM(CASE WHEN road AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS road_tag_only_per_day,
          CAST(SUM(CASE WHEN road AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS road_geometry_only_per_day,
          CAST(SUM(CASE WHEN road AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS road_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN road AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS road_none_per_day,
          -- AMENITY
          CAST(SUM(CASE WHEN amenity AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS amenity_created_per_day,
          CAST(SUM(CASE WHEN amenity AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS amenity_deleted_per_day,
          CAST(SUM(CASE WHEN amenity AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS amenity_tag_only_per_day,
          CAST(SUM(CASE WHEN amenity AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS amenity_geometry_only_per_day,
          CAST(SUM(CASE WHEN amenity AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS amenity_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN amenity AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS amenity_none_per_day,
          --BODY OF WATER
          CAST(SUM(CASE WHEN body_of_water AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS body_of_water_created_per_day,
          CAST(SUM(CASE WHEN body_of_water AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS body_of_water_deleted_per_day,
          CAST(SUM(CASE WHEN body_of_water AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS body_of_water_tag_only_per_day,
          CAST(SUM(CASE WHEN body_of_water AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS body_of_water_geometry_only_per_day,
          CAST(SUM(CASE WHEN body_of_water AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS body_of_water_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN body_of_water AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS body_of_water_none_per_day,
          --SHOP
          CAST(SUM(CASE WHEN shop AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS shop_created_per_day,
          CAST(SUM(CASE WHEN shop AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS shop_deleted_per_day,
          CAST(SUM(CASE WHEN shop AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS shop_tag_only_per_day,
          CAST(SUM(CASE WHEN shop AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS shop_geometry_only_per_day,
          CAST(SUM(CASE WHEN shop AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS shop_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN shop AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS shop_none_per_day,
          --EDUCATIONAL INSTITUTION
          CAST(SUM(CASE WHEN educational_institution AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS educational_institution_created_per_day,
          CAST(SUM(CASE WHEN educational_institution AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS educational_institution_deleted_per_day,
          CAST(SUM(CASE WHEN educational_institution AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS educational_institution_tag_only_per_day,
          CAST(SUM(CASE WHEN educational_institution AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS educational_institution_geometry_only_per_day,
          CAST(SUM(CASE WHEN educational_institution AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS educational_institution_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN educational_institution AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS educational_institution_none_per_day,
          --FINANCIAL SERVICE
          CAST(SUM(CASE WHEN financial_service AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS financial_service_created_per_day,
          CAST(SUM(CASE WHEN financial_service AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS financial_service_deleted_per_day,
          CAST(SUM(CASE WHEN financial_service AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS financial_service_tag_only_per_day,
          CAST(SUM(CASE WHEN financial_service AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS financial_service_geometry_only_per_day,
          CAST(SUM(CASE WHEN financial_service AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS financial_service_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN financial_service AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS financial_service_none_per_day,
          --HEALTHCARE FACILITY
          CAST(SUM(CASE WHEN healthcare_facility AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS healthcare_facility_created_per_day,
          CAST(SUM(CASE WHEN healthcare_facility AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS healthcare_facility_deleted_per_day,
          CAST(SUM(CASE WHEN healthcare_facility AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS healthcare_facility_tag_only_per_day,
          CAST(SUM(CASE WHEN healthcare_facility AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS healthcare_facility_geometry_only_per_day,
          CAST(SUM(CASE WHEN healthcare_facility AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS healthcare_facility_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN healthcare_facility AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS healthcare_facility_none_per_day,
          --LAND USE
          CAST(SUM(CASE WHEN land_use AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS land_use_created_per_day,
          CAST(SUM(CASE WHEN land_use AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS land_use_deleted_per_day,
          CAST(SUM(CASE WHEN land_use AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS land_use_tag_only_per_day,
          CAST(SUM(CASE WHEN land_use AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS land_use_geometry_only_per_day,
          CAST(SUM(CASE WHEN land_use AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS land_use_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN land_use AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS land_use_none_per_day,
          --PLACE
          CAST(SUM(CASE WHEN place AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS place_created_per_day,
          CAST(SUM(CASE WHEN place AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS place_deleted_per_day,
          CAST(SUM(CASE WHEN place AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS place_tag_only_per_day,
          CAST(SUM(CASE WHEN place AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS place_geometry_only_per_day,
          CAST(SUM(CASE WHEN place AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS place_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN place AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS place_none_per_day,
          --POINT OF INTEREST
          CAST(SUM(CASE WHEN point_of_interest AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS point_of_interest_created_per_day,
          CAST(SUM(CASE WHEN point_of_interest AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS point_of_interest_deleted_per_day,
          CAST(SUM(CASE WHEN point_of_interest AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS point_of_interest_tag_only_per_day,
          CAST(SUM(CASE WHEN point_of_interest AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS point_of_interest_geometry_only_per_day,
          CAST(SUM(CASE WHEN point_of_interest AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS point_of_interest_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN point_of_interest AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS point_of_interest_none_per_day,
          --SOCIAL FACILITY
          CAST(SUM(CASE WHEN social_facility AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS social_facility_created_per_day,
          CAST(SUM(CASE WHEN social_facility AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS social_facility_deleted_per_day,
          CAST(SUM(CASE WHEN social_facility AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS social_facility_tag_only_per_day,
          CAST(SUM(CASE WHEN social_facility AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS social_facility_geometry_only_per_day,
          CAST(SUM(CASE WHEN social_facility AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS social_facility_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN social_facility AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS social_facility_none_per_day,
          --WASH FACILITY
          CAST(SUM(CASE WHEN wash_facility AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS wash_facility_created_per_day,
          CAST(SUM(CASE WHEN wash_facility AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS wash_facility_deleted_per_day,
          CAST(SUM(CASE WHEN wash_facility AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS wash_facility_tag_only_per_day,
          CAST(SUM(CASE WHEN wash_facility AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS wash_facility_geometry_only_per_day,
          CAST(SUM(CASE WHEN wash_facility AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS wash_facility_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN wash_facility AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS wash_facility_none_per_day,
          --WATERWAY
          CAST(SUM(CASE WHEN waterway AND contrib_type = 'CREATION' THEN contrib_count ELSE 0 END) AS BIGINT) AS waterway_created_per_day,
          CAST(SUM(CASE WHEN waterway AND contrib_type = 'DELETION' THEN contrib_count ELSE 0 END) AS BIGINT) AS waterway_deleted_per_day,
          CAST(SUM(CASE WHEN waterway AND contrib_type = 'TAG' THEN contrib_count ELSE 0 END) AS BIGINT) AS waterway_tag_only_per_day,
          CAST(SUM(CASE WHEN waterway AND contrib_type = 'GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS waterway_geometry_only_per_day,
          CAST(SUM(CASE WHEN waterway AND contrib_type = 'TAG_GEOMETRY' THEN contrib_count ELSE 0 END) AS BIGINT) AS waterway_tag_and_geometry_per_day,
          CAST(SUM(CASE WHEN waterway AND contrib_type = '' THEN contrib_count ELSE 0 END) AS BIGINT) AS waterway_none_per_day
              FROM daily_edits
      GROUP BY user_id, country, edit_date
  ),
  changeset_editors AS (
    SELECT
      user_id,
      changeset_id,
      valid_from::DATE AS edit_date,
      MIN(editor_category) AS editor_category
    FROM osm_data_{iso_codes_str}
    GROUP BY user_id, changeset_id, edit_date
  ),

  daily_changeset_values AS (
    SELECT
      o.user_id,
      ce.edit_date,
      o.changeset_id,
      EXTRACT(EPOCH FROM o.changeset_edit_duration) AS duration_seconds,
      COUNT(*) AS count_edit_per_changeset,
      MAX(o.length_comment) AS length_comment,
      CASE WHEN ce.editor_category = 'JOSM' THEN 1 ELSE 0 END AS is_josm,
      CASE WHEN ce.editor_category = 'iD' THEN 1 ELSE 0 END AS is_id,
      CASE WHEN ce.editor_category = 'Streetcomplete' THEN 1 ELSE 0 END AS is_streetcomplete,
      CASE WHEN ce.editor_category = 'other' THEN 1 ELSE 0 END AS is_other
    FROM osm_data_{iso_codes_str} o
    JOIN changeset_editors ce
      ON o.user_id = ce.user_id AND o.changeset_id = ce.changeset_id
    GROUP BY o.user_id, edit_date, o.changeset_id, o.changeset_edit_duration, ce.editor_category
  ),

  changeset_avg AS (
    SELECT
      user_id,
      edit_date,
      AVG(count_edit_per_changeset) AS avg_edits_per_changeset_per_day,
      SUM(is_josm) AS count_josm_changesets_per_day,
      SUM(is_id) AS count_id_changesets_per_day,
      SUM(is_streetcomplete) AS count_streetcomplete_changesets_per_day,
      SUM(is_other) AS count_other_id_changesets_per_day,
      COUNT(*) AS total_changesets_per_day,
      SUM(length_comment) AS total_comment_length_per_day,
      SUM(duration_seconds) AS total_changeset_duration_per_day_in_sec,
    FROM daily_changeset_values
    GROUP BY user_id, edit_date
  ),
  aggregated_tag_changes AS (
    SELECT
      user_id,
      valid_from::DATE AS edit_date,
      SUM(length_removed_keys) AS total_removed_keys_per_day,
      SUM(length_new_keys) AS total_new_keys_per_day,

    FROM osm_data_{iso_codes_str}
    GROUP BY user_id, edit_date
  ),
  removed_tags_flat AS (
    SELECT
      user_id,
      valid_from::DATE AS edit_date,
      UNNEST(removed_feature_categories) AS category
    FROM osm_data_{iso_codes_str}
  ),
  new_tags_flat AS (
    SELECT
      user_id,
      valid_from::DATE AS edit_date,
      UNNEST(new_feature_categories) AS category
    FROM osm_data_{iso_codes_str}
  ),
  removed_category_counts AS (
    SELECT
      user_id,
      edit_date,
      SUM(CASE WHEN category = 'building' THEN 1 ELSE 0 END) AS removed_key_building_per_day,
      SUM(CASE WHEN category = 'road' THEN 1 ELSE 0 END) AS removed_key_road_per_day,
      SUM(CASE WHEN category = 'amenity' THEN 1 ELSE 0 END) AS removed_key_amenity_per_day,
      SUM(CASE WHEN category = 'body_of_water' THEN 1 ELSE 0 END) AS removed_key_body_of_water_per_day,
      SUM(CASE WHEN category = 'shop' THEN 1 ELSE 0 END) AS removed_key_shop_per_day,
      SUM(CASE WHEN category = 'educational_institution' THEN 1 ELSE 0 END) AS removed_key_educational_institution_per_day,
      SUM(CASE WHEN category = 'financial_service' THEN 1 ELSE 0 END) AS removed_key_financial_service_per_day,
      SUM(CASE WHEN category = 'healthcare_facility' THEN 1 ELSE 0 END) AS removed_key_healthcare_facility_per_day,
      SUM(CASE WHEN category = 'land_use' THEN 1 ELSE 0 END) AS removed_key_land_use_per_day,
      SUM(CASE WHEN category = 'place' THEN 1 ELSE 0 END) AS removed_key_place_per_day,
      SUM(CASE WHEN category = 'point_of_interest' THEN 1 ELSE 0 END) AS removed_key_poi_per_day,
      SUM(CASE WHEN category = 'social_facility' THEN 1 ELSE 0 END) AS removed_key_social_facility_per_day,
      SUM(CASE WHEN category = 'wash_facility' THEN 1 ELSE 0 END) AS removed_key_wash_facility_per_day,
      SUM(CASE WHEN category = 'waterway' THEN 1 ELSE 0 END) AS removed_key_waterway_per_day,
      SUM(CASE WHEN category = 'other' THEN 1 ELSE 0 END) AS removed_key_other_per_day
    FROM removed_tags_flat
    GROUP BY user_id, edit_date
  ),
  new_category_counts AS (
    SELECT
      user_id,
      edit_date,

      -- Count new keys per category:
      SUM(CASE WHEN category = 'building' THEN 1 ELSE 0 END) AS new_key_building_per_day,
      SUM(CASE WHEN category = 'road' THEN 1 ELSE 0 END) AS new_key_road_per_day,
      SUM(CASE WHEN category = 'amenity' THEN 1 ELSE 0 END) AS new_key_amenity_per_day,
      SUM(CASE WHEN category = 'body_of_water' THEN 1 ELSE 0 END) AS new_key_body_of_water_per_day,
      SUM(CASE WHEN category = 'shop' THEN 1 ELSE 0 END) AS new_key_shop_per_day,
      SUM(CASE WHEN category = 'educational_institution' THEN 1 ELSE 0 END) AS new_key_educational_institution_per_day,
      SUM(CASE WHEN category = 'financial_service' THEN 1 ELSE 0 END) AS new_key_financial_service_per_day,
      SUM(CASE WHEN category = 'healthcare_facility' THEN 1 ELSE 0 END) AS new_key_healthcare_facility_per_day,
      SUM(CASE WHEN category = 'land_use' THEN 1 ELSE 0 END) AS new_key_land_use_per_day,
      SUM(CASE WHEN category = 'place' THEN 1 ELSE 0 END) AS new_key_place_per_day,
      SUM(CASE WHEN category = 'point_of_interest' THEN 1 ELSE 0 END) AS new_key_poi_per_day,
      SUM(CASE WHEN category = 'social_facility' THEN 1 ELSE 0 END) AS new_key_social_facility_per_day,
      SUM(CASE WHEN category = 'wash_facility' THEN 1 ELSE 0 END) AS new_key_wash_facility_per_day,
      SUM(CASE WHEN category = 'waterway' THEN 1 ELSE 0 END) AS new_key_waterway_per_day,
      SUM(CASE WHEN category = 'other' THEN 1 ELSE 0 END) AS new_key_other_per_day

    FROM new_tags_flat
    GROUP BY user_id, edit_date
  )


  SELECT 
    s.*,
    c.total_comment_length_per_day,
    c.avg_edits_per_changeset_per_day,
    c.count_josm_changesets_per_day,
    c.count_id_changesets_per_day,
    c.count_streetcomplete_changesets_per_day,
    c.count_other_id_changesets_per_day,
    c.total_changesets_per_day,
    c.total_changeset_duration_per_day_in_sec,
    t.total_new_keys_per_day,
    t.total_removed_keys_per_day,
    n.* exclude(user_id, edit_date),
    r.* exclude(user_id, edit_date),


  FROM summed s
  LEFT JOIN changeset_avg c
    ON s.user_id = c.user_id AND s.edit_date = c.edit_date
  LEFT JOIN aggregated_tag_changes t
    ON s.user_id = t.user_id AND s.edit_date = t.edit_date
  LEFT JOIN new_category_counts n
    ON s.user_id = n.user_id AND s.edit_date = n.edit_date
  LEFT JOIN removed_category_counts r
    ON s.user_id = r.user_id AND s.edit_date = r.edit_date;
  """

def build_drop_enhancement_sql(iso_code: str) -> str:
    return f"DROP TABLE IF EXISTS osm_data_{iso_code};"


def build_export_parquet_sql(iso_code: str) -> str:
    export_path = f"{EXPORT_DAILY_PREFIX}_{iso_code}.parquet".replace("\\", "/")
    return f"COPY daily_per_user_{iso_code} TO '{export_path}' (FORMAT PARQUET);"



# ============= 3. MAIN PER-COUNTRY PIPELINE ============

def run_country(iso_code: str, access_key: str, secret_key: str) -> None:
    iso_code = iso_code.upper()
    print(f"\n=== Running pipeline for {iso_code} ===")

    condition = build_iso_condition(iso_code)
    bbox = get_country_bbox(iso_code, WORLD_GDF)
    print("Condition:", condition)
    print("BBOX     :", bbox)

    secret_sql = build_secret_sql(access_key, secret_key)
    enhancement_sql = build_enhancement_query(iso_code, condition, bbox)
    daily_user_sql = build_daily_user_query(iso_code)
    drop_sql = build_drop_enhancement_sql(iso_code)
    export_sql = build_export_parquet_sql(iso_code)

    # temp SQL file
    sql_file = BASE_DIR / "sql" / "temp_query.sql"
    sql_file.parent.mkdir(parents=True, exist_ok=True)

    with sql_file.open("w", encoding="utf-8") as f:
        f.write(secret_sql + "\n")
        f.write(MAX_MEMORY_SQL + "\n")
        f.write(enhancement_sql + "\n")
        f.write(daily_user_sql + "\n")
        f.write(drop_sql + "\n")
        f.write(export_sql + "\n")

    duckdb_bin = shutil.which("duckdb") or shutil.which("duckdb.exe")
    if not duckdb_bin:
        raise FileNotFoundError("DuckDB CLI not found (duckdb/duckdb.exe).")

    print("Using DuckDB CLI:", duckdb_bin)
    with sql_file.open("rb") as f:
        subprocess.run([duckdb_bin, ":memory:"], stdin=f, check=True)


# ============= 4. CLI ENTRYPOINT ========================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <ISO_A3_CODE> [<ISO_A3_CODE_2> ...]")
        sys.exit(1)

    print(f"DuckDB MAX_MEMORY: {duckdb_ram_gb}GB")
    print(f"DuckDB THREADS   : {num_threads}")

    access_key = input("Enter your S3 Access Key: ")
    secret_key = input("Enter your S3 Secret Key: ")

    iso_codes = [arg.upper() for arg in sys.argv[1:]]
    for code in iso_codes:
        run_country(code, access_key, secret_key)

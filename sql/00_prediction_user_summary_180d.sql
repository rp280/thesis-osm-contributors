----- FOR PREDICTION (First 180 days of each user after first edit are considered) -----
copy (
WITH
first_edit_per_user AS (
  SELECT 
    user_id, 
    MIN(edit_date) AS first_edit,
	MAX(edit_date) AS full_last_edit
  FROM read_parquet('{input_pattern}')
  GROUP BY user_id
  HAVING MIN(edit_date) <= DATE '2025-07-28' - INTERVAL 180 DAY
),
user_edits_180 AS (
  SELECT 
    r.*
  FROM read_parquet('{input_pattern}') AS r
  JOIN first_edit_per_user f ON r.user_id = f.user_id
  WHERE r.edit_date <= DATE_ADD(f.first_edit, INTERVAL 180 DAY)
),
per_country_edits AS (
  SELECT
    user_id,
    country,
    MIN(edit_date) AS min_date,
	  MAX(edit_date) AS max_date,
    SUM(edit_count) AS sum_edits,
    COUNT(DISTINCT edit_date::DATE) AS days_editing_country,
    SUM(count_created_per_day) AS sum_created,
    SUM(count_deleted_per_day) AS sum_deleted,
    SUM(count_tag_only_per_day) AS sum_tagged,
    SUM(count_geom_only_per_day) AS sum_geometry,
    SUM(count_tag_and_geom_per_day) AS sum_geometry_tag,
    SUM(count_none_per_day) AS sum_none,
    SUM(count_building_per_day) AS sum_building,
    SUM(count_road_per_day) AS sum_road,
    SUM(count_amenity_per_day) AS sum_amenity,
    SUM(count_body_of_water_per_day) AS sum_body_of_water,
    SUM(count_shop_per_day) AS sum_shop,
    SUM(count_educational_institution_per_day) AS sum_educational_institution,
    SUM(count_financial_service_per_day) AS sum_financial_service,
    SUM(count_healthcare_facility_per_day) AS sum_healthcare_facility,
    SUM(count_land_use_per_day) AS sum_land_use,
    SUM(count_place_per_day) AS sum_place,
    SUM(count_point_of_interest_per_day) AS sum_point_of_interest,
    SUM(count_social_facility_per_day) AS sum_social_facility,
    SUM(count_wash_facility_per_day) AS sum_wash_facility,
    SUM(count_waterway_per_day) AS sum_waterway,
    SUM(count_node_per_day) AS total_nodes_per_country,
    SUM(count_way_per_day) AS total_ways_per_country,
    SUM(count_relation_per_day) AS total_relations_per_country,
    SUM(total_new_keys_per_day) AS total_new_keys_per_country,
    SUM(total_removed_keys_per_day) AS total_deleted_keys_per_country,
    SUM(total_changesets_per_day) AS total_changesets_per_country,
    SUM(total_changeset_duration_per_day_in_sec) AS total_changeset_duration_per_country_in_sec,
    SUM(count_josm_changesets_per_day) AS total_josm_changesets_per_country,
    SUM(count_id_changesets_per_day) AS total_id_changesets_per_country,
    SUM(count_streetcomplete_changesets_per_day) AS total_streetcomplete_changesets_per_country,
    SUM(count_other_id_changesets_per_day) AS total_other_id_changesets_per_country,
    SUM(total_comment_length_per_day) AS total_comment_length_per_country,
    SUM(total_days_valid) AS total_days_valid_per_country
  FROM user_edits_180
  GROUP BY user_id, country
),
total_edits AS (
  SELECT
    user_id,
    SUM(sum_edits) AS total_edits
  FROM per_country_edits
  GROUP BY user_id
 ),
 true_edit_days AS (
  SELECT
    user_id,
    COUNT(DISTINCT edit_date::DATE) AS true_edit_days
  FROM user_edits_180
  GROUP BY user_id
),
pause_after_first_edit AS (
  WITH user_edits_ranked AS (
    SELECT
      user_id,
      edit_date::DATE AS edit_day,
      ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY edit_date::DATE) AS rn
    FROM user_edits_180
  )
  SELECT
    user_id,
    MAX(CASE WHEN rn = 1 THEN edit_day ELSE NULL END) AS first_day,
    MAX(CASE WHEN rn = 2 THEN edit_day ELSE NULL END) AS second_day,
    DATE_DIFF('day',
      MAX(CASE WHEN rn = 1 THEN edit_day ELSE NULL END),
      MAX(CASE WHEN rn = 2 THEN edit_day ELSE NULL END)
    ) AS pause_after_first_edit
  FROM user_edits_ranked
  WHERE rn <= 2
  GROUP BY user_id
),
burstiness AS (
  WITH user_edits AS (
    SELECT
      user_id,
      edit_date::DATE AS day
    FROM user_edits_180
    GROUP BY user_id, day
  ),
  edit_diffs AS (
    SELECT
      user_id,
      day,
      day - LAG(day) OVER (PARTITION BY user_id ORDER BY day) AS diff_days
    FROM user_edits
  )
  SELECT
    user_id,
    STDDEV(diff_days) AS burstiness
  FROM edit_diffs
  WHERE diff_days IS NOT NULL
  GROUP BY user_id
),
top_country_per_user AS (
  SELECT user_id, country AS top_country
  FROM (
    SELECT
      user_id,
      country,
      ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY SUM(edit_count) DESC) AS rn
    FROM user_edits_180
    GROUP BY user_id, country
  )
  WHERE rn = 1
),
country_probs AS (
  SELECT
    p.user_id,
    1.0 * p.sum_edits / NULLIF(t.total_edits, 0) AS p_i
  FROM per_country_edits p
  JOIN total_edits t USING (user_id)
),
entropy_calc AS (
  SELECT
    user_id,
    ROUND(-SUM(p_i * LOG(p_i) / LOG(2)), 4) AS entropy_country_distribution
  FROM country_probs
  GROUP BY user_id
),
        weekly AS (
  SELECT user_id, DATE_TRUNC('week', edit_date) AS wk
  FROM user_edits_180
  GROUP BY user_id, wk
),
weekly_stats AS (
  SELECT
    user_id,
    COUNT(*) AS active_weeks_26,
    COUNT(*) * 1.0 / 26 AS active_week_ratio
  FROM weekly
  GROUP BY user_id
),
cum AS (
  SELECT
    user_id, edit_date,
    SUM(edit_count) OVER (PARTITION BY user_id ORDER BY edit_date ROWS UNBOUNDED PRECEDING) AS cum_edits
  FROM user_edits_180
),
milestones AS (
  SELECT
    user_id,
    MIN(CASE WHEN cum_edits >= 50  THEN edit_date END) AS day_50,
    MIN(CASE WHEN cum_edits >= 100 THEN edit_date END) AS day_100
  FROM cum
  GROUP BY user_id
),
milestone_features AS (
  SELECT
    m.user_id,
    DATE_DIFF('day', f.first_edit, m.day_50)  AS days_to_50,
    DATE_DIFF('day', f.first_edit, m.day_100) AS days_to_100
  FROM milestones m
  JOIN first_edit_per_user f USING (user_id)
),
early_windows AS (
  SELECT
    r.user_id,
    SUM(CASE WHEN r.edit_date <= f.first_edit + INTERVAL 7  DAY THEN r.edit_count ELSE 0 END) AS edits_7d,
    SUM(CASE WHEN r.edit_date <= f.first_edit + INTERVAL 30 DAY THEN r.edit_count ELSE 0 END) AS edits_30d,
    COUNT(DISTINCT CASE WHEN r.edit_date <= f.first_edit + INTERVAL 7  DAY THEN r.edit_date END)  AS days_7d,
    COUNT(DISTINCT CASE WHEN r.edit_date <= f.first_edit + INTERVAL 30 DAY THEN r.edit_date END) AS days_30d
  FROM user_edits_180 r
  JOIN first_edit_per_user f USING (user_id)
  GROUP BY r.user_id
),
early_ratios AS (
  SELECT
    user_id,
    edits_7d * 1.0 / NULLIF(edits_30d,0)  AS frontload_ratio,    -- proportion of edits in the first 30 days that occurred in the first 7 days (how strongly concentrated in week 1)
    days_7d  * 1.0 / NULLIF(days_30d,0)   AS early_days_ratio    -- Proportion of active days (at least 1 day editing) that occurred in the first 7 days
  FROM early_windows
), 
span_180 AS (
  SELECT
    user_id,
    MIN(edit_date) AS first_in_180,
    MAX(edit_date) AS last_in_180
  FROM user_edits_180
  GROUP BY user_id
),
user_summary AS (
  SELECT
    user_id,
    MIN(min_date) AS first_edit,
    MAX(max_date) AS last_edit,
    COUNT(DISTINCT country) AS number_of_countries,
    MAX(sum_edits) AS max_country_edits,
    SUM(sum_edits) AS total_edits,
    SUM(days_editing_country) AS edit_days,
    SUM(sum_created) AS total_created,
    SUM(sum_deleted) AS total_deleted,
    SUM(sum_tagged) AS total_tagged,
    SUM(sum_geometry) AS total_geometry,
    SUM(sum_geometry_tag) AS total_geometry_tag,
    SUM(sum_none) AS total_none,
    SUM(sum_building) AS total_building,
    SUM(sum_road) AS total_road,
    SUM(sum_amenity) AS total_amenity,
    SUM(sum_body_of_water) AS total_body_of_water,
    SUM(sum_shop) AS total_shop,
    SUM(sum_educational_institution) AS total_educational_institution,
    SUM(sum_financial_service) AS total_financial_service,
    SUM(sum_healthcare_facility) AS total_healthcare_facility,
    SUM(sum_land_use) AS total_land_use,
    SUM(sum_place) AS total_place,
    SUM(sum_point_of_interest) AS total_point_of_interest,
    SUM(sum_social_facility) AS total_social_facility,
    SUM(sum_wash_facility) AS total_wash_facility,
    SUM(sum_waterway) AS total_waterway,
    
    (
      (CASE WHEN SUM(sum_building) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_road) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_amenity) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_body_of_water) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_shop) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_educational_institution) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_financial_service) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_healthcare_facility) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_land_use) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_place) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_point_of_interest) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_social_facility) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_wash_facility) > 0 THEN 1 ELSE 0 END) +
      (CASE WHEN SUM(sum_waterway) > 0 THEN 1 ELSE 0 END)
    ) AS feature_category_count,
    (
      SUM(sum_building) + SUM(sum_road) + SUM(sum_amenity) +
      SUM(sum_body_of_water) + SUM(sum_shop) +
      SUM(sum_educational_institution) + SUM(sum_financial_service) +
      SUM(sum_healthcare_facility) + SUM(sum_land_use) + SUM(sum_place) +
      SUM(sum_point_of_interest) + SUM(sum_social_facility) +
      SUM(sum_wash_facility) + SUM(sum_waterway)
    ) AS total_featuretype_edits,
    GREATEST(
      SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
      SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
      SUM(sum_financial_service), SUM(sum_healthcare_facility),
      SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
      SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
    ) AS max_featuretype_count,
    CASE 
		  WHEN total_featuretype_edits = 0 THEN 'none'
		  WHEN GREATEST(
				SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
				SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
				SUM(sum_financial_service), SUM(sum_healthcare_facility),
				SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
				SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
			  ) = SUM(sum_building) THEN 'building'
		  WHEN GREATEST(SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
				SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
				SUM(sum_financial_service), SUM(sum_healthcare_facility),
				SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
				SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)) = SUM(sum_road) THEN 'road'
		  WHEN GREATEST(SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
				SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
				SUM(sum_financial_service), SUM(sum_healthcare_facility),
				SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
				SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)) = SUM(sum_amenity) THEN 'amenity'
		   WHEN GREATEST(SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
				SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
				SUM(sum_financial_service), SUM(sum_healthcare_facility),
				SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
				SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)) = SUM(sum_body_of_water) THEN 'body_of_water'
				WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_shop) THEN 'shop'

				  WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_educational_institution) THEN 'educational_institution'

				  WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_financial_service) THEN 'financial_service'

				  WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_healthcare_facility) THEN 'healthcare_facility'

				  WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_land_use) THEN 'land_use'

				  WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_place) THEN 'place'

				  WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_point_of_interest) THEN 'point_of_interest'

				  WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_social_facility) THEN 'social_facility'

				  WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_wash_facility) THEN 'wash_facility'

				  WHEN GREATEST(
					SUM(sum_building), SUM(sum_road), SUM(sum_amenity),
					SUM(sum_body_of_water), SUM(sum_shop), SUM(sum_educational_institution),
					SUM(sum_financial_service), SUM(sum_healthcare_facility),
					SUM(sum_land_use), SUM(sum_place), SUM(sum_point_of_interest),
					SUM(sum_social_facility), SUM(sum_wash_facility), SUM(sum_waterway)
				  ) = SUM(sum_waterway) THEN 'waterway'

		  ELSE 'unknown'
		END AS top_feature_type_name,
		SUM(total_nodes_per_country) AS total_nodes,
    SUM(total_ways_per_country) AS total_ways,
    SUM(total_relations_per_country) AS total_relations,
    SUM(total_new_keys_per_country) AS total_new_keys,
    SUM(total_deleted_keys_per_country) AS total_deleted_keys,
    SUM(total_changesets_per_country) AS total_changesets,
    SUM(total_changeset_duration_per_country_in_sec) AS total_changeset_duration_in_sec,
    SUM(total_josm_changesets_per_country) AS total_josm_changesets,
    SUM(total_id_changesets_per_country) AS total_id_changesets,
    SUM(total_streetcomplete_changesets_per_country) AS total_streetcomplete_changesets,
    SUM(total_other_id_changesets_per_country) AS total_other_id_changesets,
    SUM(total_comment_length_per_country) AS total_comment_length,
    SUM(total_days_valid_per_country) AS total_days_valid,
  FROM per_country_edits
  GROUP BY user_id
)
SELECT
  -- This select represents the data for the beginning 6 Month of each user
  us.user_id,
  --temporal metrics
  us.first_edit,
  fepu.full_last_edit,
  EXTRACT(YEAR FROM us.first_edit) AS first_edit_year,
  ted.true_edit_days,
  DATE_DIFF('day', us.first_edit, us.last_edit) AS active_duration,
  ted.true_edit_days * 1.0 /180 AS activity_ratio_true,
  us.total_edits * 1.0 / NULLIF(ted.true_edit_days, 0) AS edits_per_day_true,
  DATE_DIFF('day', us.first_edit, us.last_edit) * 1.0 / NULLIF(ted.true_edit_days - 1, 0) AS mean_days_between_edits_true,
  pause.pause_after_first_edit,
  EXTRACT(MONTH FROM us.first_edit) AS first_edit_month,
  --spatial metrics
  us.max_country_edits * 1.0 / NULLIF(us.total_edits, 0) AS top_country_ratio,
  ec.entropy_country_distribution,
  (1 - (us.max_country_edits * 1.0 / NULLIF(us.total_edits, 0))) * LOG(us.total_edits + 1) AS diversity_score,
  us.number_of_countries,
  us.max_country_edits,
  tcp.top_country,
  --edit type (ratios)
  us.total_created * 1.0 / NULLIF(us.total_edits, 0) AS created_ratio,
  us.total_deleted * 1.0 / NULLIF(us.total_edits, 0) AS deleted_ratio,
  us.total_tagged * 1.0 / NULLIF(us.total_edits, 0) AS tag_ratio,
  us.total_geometry * 1.0 / NULLIF(us.total_edits, 0) AS geometry_ratio,
  us.total_none * 1.0 / NULLIF(us.total_edits, 0) AS none_ratio,
  total_edits, 
  b.burstiness,
  --feature type ratio
  us.total_building * 1.0 / NULLIF(us.total_edits, 0) AS building_ratio,
  us.total_road * 1.0 / NULLIF(us.total_edits, 0) AS road_ratio,
  us.total_amenity * 1.0 / NULLIF(us.total_edits, 0) AS amenity_ratio,
  us.total_body_of_water * 1.0 / NULLIF(us.total_edits, 0) AS body_of_water_ratio,
  us.total_shop * 1.0 / NULLIF(us.total_edits, 0) AS shop_ratio,
  us.total_educational_institution * 1.0 / NULLIF(us.total_edits, 0) AS educational_institution_ratio,
  us.total_financial_service * 1.0 / NULLIF(us.total_edits, 0) AS financial_service_ratio,
  us.total_healthcare_facility * 1.0 / NULLIF(us.total_edits, 0) AS healthcare_facility_ratio,
  us.total_land_use * 1.0 / NULLIF(us.total_edits, 0) AS land_use_ratio,
  us.total_place * 1.0 / NULLIF(us.total_edits, 0) AS place_ratio,
  us.total_point_of_interest * 1.0 / NULLIF(us.total_edits, 0) AS point_of_interest_ratio,
  us.total_social_facility * 1.0 / NULLIF(us.total_edits, 0) AS social_facility_ratio,
  us.total_wash_facility * 1.0 / NULLIF(us.total_edits, 0) AS wash_facility_ratio,
  us.total_waterway * 1.0 / NULLIF(us.total_edits, 0) AS waterway_ratio,
  --feature type edit
  us.feature_category_count,
  us.max_featuretype_count * 1.0 / NULLIF(us.total_featuretype_edits, 0) AS top_feature_ratio,
  us.top_feature_type_name,
  -- new variables
  us.total_nodes * 1.0 / NULLIF(us.total_edits, 0) AS node_ratio,
  us.total_ways * 1.0 / NULLIF(us.total_edits, 0) AS way_ratio,
  us.total_relations * 1.0 / NULLIF(us.total_edits, 0) AS relation_ratio,
  us.total_geometry_tag * 1.0 / NULLIF(us.total_edits, 0) AS geometry_tag_ratio,
  us.total_new_keys * 1.0 / NULLIF(us.total_edits, 0) AS new_keys_ratio,
  us.total_deleted_keys * 1.0 / NULLIF(us.total_edits, 0) AS deleted_keys_ratio,
  (us.total_new_keys + us.total_deleted_keys) / NULLIF(us.total_changesets, 0) AS keys_per_changeset_ratio,
  us.total_edits * 1.0 / NULLIF(us.total_changesets, 0) AS edits_per_changeset_ratio,
  us.total_changeset_duration_in_sec * 1.0 / NULLIF(us.total_changesets, 0) AS changeset_duration_ratio_in_sec,
  us.total_josm_changesets * 1.0 / NULLIF(us.total_changesets, 0) AS josm_changeset_ratio,
  us.total_id_changesets * 1.0 / NULLIF(us.total_changesets, 0) AS id_changeset_ratio,
  us.total_streetcomplete_changesets * 1.0 / NULLIF(us.total_changesets, 0) AS streetcomplete_changeset_ratio,
  us.total_other_id_changesets * 1.0 / NULLIF(us.total_changesets, 0) AS other_id_changeset_ratio,
  us.total_comment_length * 1.0 / NULLIF(us.total_changesets, 0) AS comment_length_ratio,
  us.total_changesets * 1.0 / NULLIF(ted.true_edit_days, 0) AS changesets_per_edit_day,
  us.total_days_valid * 1.0 / NULLIF(us.total_edits, 0) AS avg_days_valid_per_edit,
  ws.active_weeks_26,
  mf.days_to_50,
  mf.days_to_100,
  --early ratios
  er.frontload_ratio,
  er.early_days_ratio,
  ted.true_edit_days * 1.0 / NULLIF(CASE WHEN ted.true_edit_days > 0 THEN DATE_DIFF('day', sp.first_in_180, sp.last_in_180)+1 ELSE 0 END, 0) AS activity_density,
  --target variable (left before 6 month)
  CASE
	WHEN DATE_DIFF('day', fepu.first_edit, fepu.full_last_edit) < 180 THEN TRUE
  ELSE FALSE
	END AS left_early

FROM user_summary us
LEFT JOIN burstiness b ON us.user_id = b.user_id
LEFT JOIN entropy_calc ec ON us.user_id = ec.user_id
LEFT JOIN top_country_per_user tcp ON us.user_id = tcp.user_id
LEFT JOIN true_edit_days ted ON us.user_id = ted.user_id
LEFT JOIN pause_after_first_edit pause ON us.user_id = pause.user_id
LEFT JOIN first_edit_per_user fepu ON us.user_id = fepu.user_id
LEFT JOIN weekly_stats ws ON us.user_id = ws.user_id
LEFT JOIN milestone_features  mf  ON us.user_id = mf.user_id
LEFT JOIN early_ratios        er  ON us.user_id = er.user_id
LEFT JOIN span_180        sp  ON us.user_id = sp.user_id

ORDER BY us.user_id
) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
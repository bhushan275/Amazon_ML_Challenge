# Data Schemas & Pipeline Contracts

## Normalized Source Schema (`stubs/normalized_source*.tsv`)

Every normalized source dataset (`normalized_source1.tsv`, `normalized_source2.tsv`, `normalized_source3.tsv`) MUST adhere to the tab-separated (TSV) column schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `entity_id` | `str` | Unique entity identifier (e.g. `S1-101`, `S2-201`, `S3-301`) |
| `name_norm` | `str` | Cleaned business name (unidecode, lowercase, legal suffixes stripped) |
| `address_norm` | `str` | Full normalized address string with expanded abbreviations |
| `street` | `str` | Standardized street component |
| `city` | `str` | Standardized city component |
| `state` | `str` | Standardized state/province component |
| `postal_code` | `str` | Extracted ZIP/PIN code |
| `country` | `str` | Canonical ISO-ish country code/name (e.g. `US`, `IN`, `FR`) |

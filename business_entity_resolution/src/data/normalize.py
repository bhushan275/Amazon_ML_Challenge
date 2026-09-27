"""
Text normalization, cleaning, and standardize preprocessing routines for entity attributes.
"""

import re
from typing import Dict, Optional, Union, Tuple
from unidecode import unidecode
import pandas as pd


# Controlled tuple of legal business suffixes to strip from business names
SUFFIXES: Tuple[str, ...] = (
    "inc", "incorporated", "corp", "corporation", "llc", "l.l.c.", "l l c",
    "ltd", "limited", "pvt", "private", "gmbh", "sa", "sarl", "co", "company",
    "plc", "bv", "b.v.", "nv", "ag", "spa", "s.p.a.", "sociedad anonima",
    "s.a.", "s.a", "s.a.r.l.", "s/a", "holding", "holdings", "group", "services",
    "enterprises", "enterprise", "solutions", "international", "int", "intl",
    "pvt ltd", "private limited", "co ltd", "co. ltd.", "llp", "l.l.p."
)

# Address term abbreviation expansion mapping
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "st": "street",
    "st.": "street",
    "str": "street",
    "rd": "road",
    "rd.": "road",
    "ave": "avenue",
    "ave.": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ln": "lane",
    "ln.": "lane",
    "ct": "court",
    "ct.": "court",
    "pkwy": "parkway",
    "pkwy.": "parkway",
    "sq": "square",
    "ste": "suite",
    "apt": "apartment",
    "hwy": "highway",
    "pl": "place",
    "fl": "floor",
    "bldg": "building",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

# Known country alias lookup table for canonical normalization
COUNTRY_ALIASES: Dict[str, str] = {
    "us": "US",
    "usa": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "united states": "US",
    "united states of america": "US",
    "america": "US",
    "in": "IN",
    "ind": "IN",
    "india": "IN",
    "bharat": "IN",
    "gb": "GB",
    "gbr": "GB",
    "uk": "GB",
    "u.k.": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "ca": "CA",
    "can": "CA",
    "canada": "CA",
    "fr": "FR",
    "fra": "FR",
    "france": "FR",
    "republique francaise": "FR",
    "de": "DE",
    "deu": "DE",
    "germany": "DE",
    "deutschland": "DE",
    "jp": "JP",
    "jpn": "JP",
    "japan": "JP",
    "nippon": "JP",
    "cn": "CN",
    "chn": "CN",
    "china": "CN",
    "au": "AU",
    "aus": "AU",
    "australia": "AU",
    "br": "BR",
    "bra": "BR",
    "brazil": "BR",
    "brasil": "BR",
    "mx": "MX",
    "mex": "MX",
    "mexico": "MX",
    "es": "ES",
    "esp": "ES",
    "spain": "ES",
    "espana": "ES",
    "it": "IT",
    "ita": "IT",
    "italy": "IT",
    "italia": "IT",
    "nl": "NL",
    "nld": "NL",
    "netherlands": "NL",
    "holland": "NL",
    "sg": "SG",
    "sgp": "SG",
    "singapore": "SG",
    "ae": "AE",
    "are": "AE",
    "uae": "AE",
    "united arab emirates": "AE",
}


def normalize_name(name: Optional[str]) -> str:
    """Normalize business entity name.

    Performs unidecode transliteration, lowercasing, punctuation stripping,
    and strips legal business suffixes defined in SUFFIXES.

    Args:
        name: Raw business entity name string.

    Returns:
        str: Cleaned and normalized name string.
    """
    if name is None or pd.isna(name):
        return ""

    text = unidecode(str(name)).lower().strip()
    if not text:
        return ""

    # Remove dots from acronyms and abbreviations (e.g. A.C.M.E. -> ACME, S.A.R.L. -> SARL, Inc. -> Inc)
    for _ in range(3):
        text = re.sub(r"\b([a-z]+)\.", r"\1", text)

    # Replace remaining non-alphanumeric characters with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()

    # Strip trailing legal suffixes recursively
    changed = True
    while changed and tokens:
        changed = False
        if len(tokens) >= 2:
            two_word = f"{tokens[-2]} {tokens[-1]}"
            if two_word in SUFFIXES:
                tokens = tokens[:-2]
                changed = True
                continue
        if tokens and tokens[-1] in SUFFIXES:
            tokens = tokens[:-1]
            changed = True

    return " ".join(tokens).strip()



def normalize_address(address: Optional[str]) -> Dict[str, str]:
    """Parse address into street, city, state, postal code, and full address_norm string.

    Applies abbreviation expansion (e.g. St->Street, Rd->Road) and defensive parsing.

    Args:
        address: Raw address string.

    Returns:
        dict: Dictionary containing keys:
            ['street', 'city', 'state', 'postal_code', 'address_norm']
    """
    empty_result = {
        "street": "",
        "city": "",
        "state": "",
        "postal_code": "",
        "address_norm": ""
    }

    if address is None or pd.isna(address):
        return empty_result

    raw_str = unidecode(str(address)).strip()
    if not raw_str:
        return empty_result

    # Postal code extraction regex
    postal_code = ""
    postal_match = re.search(r"\b(\d{5}(?:-\d{4})?|\d{6})\b", raw_str)
    if postal_match:
        postal_code = postal_match.group(1)
        raw_str = raw_str.replace(postal_match.group(0), "").strip()

    # Split into comma-separated components
    parts = [p.strip() for p in raw_str.split(",") if p.strip()]

    def expand_abbreviations(text: str) -> str:
        words = text.split()
        expanded = []
        for w in words:
            clean_w = re.sub(r"[^\w\.]", "", w).lower()
            expanded.append(ADDRESS_ABBREVIATIONS.get(clean_w, w.lower()))
        return " ".join(expanded)

    street, city, state = "", "", ""

    if len(parts) == 1:
        street = parts[0]
    elif len(parts) == 2:
        street, city = parts[0], parts[1]
    elif len(parts) >= 3:
        # Check if first part looks like a state code (e.g. "OH, Columbus, 5559 Orville Avenue")
        if len(parts[0]) == 2 and parts[0].isalpha() and parts[0].isupper():
            state = parts[0]
            city = parts[1]
            street = ", ".join(parts[2:])
        else:
            sub_unit_keywords = {"ste", "ste.", "suite", "apt", "apt.", "apartment", "bldg", "bldg.", "building", "unit", "fl", "floor"}
            has_sub_unit = any(w.lower() in sub_unit_keywords for p in parts for w in p.split())
            if has_sub_unit and len(parts) <= 3:
                street = ", ".join(parts)
            else:
                street = ", ".join(parts[:-2])
                city = parts[-2]
                state = parts[-1]

    street_norm = expand_abbreviations(street) if street else ""
    city_norm = expand_abbreviations(city) if city else ""
    state_norm = expand_abbreviations(state) if state else ""

    norm_components = [c for c in [street_norm, city_norm, state_norm, postal_code] if c]
    address_norm = ", ".join(norm_components)

    return {
        "street": street_norm,
        "city": city_norm,
        "state": state_norm,
        "postal_code": postal_code,
        "address_norm": address_norm,
    }



def normalize_country(country: Optional[str]) -> str:
    """Normalize raw country string to canonical ISO-ish country code/name.

    Generalizes to unseen countries via generic cleanup and known alias fallback table.

    Args:
        country: Raw country string.

    Returns:
        str: Standardized uppercase country code/name (e.g. 'US', 'IN', 'FRANCE').
    """
    if country is None or pd.isna(country):
        return ""

    clean_str = unidecode(str(country)).strip()
    if not clean_str:
        return ""

    lookup_key = clean_str.lower().strip()

    if lookup_key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[lookup_key]

    # Generalization fallback for unlisted countries (e.g. France -> FRANCE or FR)
    clean_alpha = re.sub(r"[^\w\s]", "", clean_str).strip()
    return clean_alpha.upper()


def normalize_record(row: Union[pd.Series, Dict]) -> Dict[str, str]:
    """Normalize an individual entity record into standardized schema dictionary.

    Args:
        row: Row object (pd.Series or dict) containing raw entity attributes.

    Returns:
        dict: Normalized record with keys:
            ['entity_id', 'name_norm', 'address_norm', 'street', 'city', 'state', 'postal_code', 'country']
    """
    if isinstance(row, pd.Series):
        row_dict = row.to_dict()
    elif isinstance(row, dict):
        row_dict = row
    else:
        row_dict = dict(row)

    entity_id = row_dict.get("entity_id") or row_dict.get("source1_entity_id") or row_dict.get("id") or ""
    raw_name = row_dict.get("business_name") or row_dict.get("name") or ""
    raw_address = row_dict.get("business_address") or row_dict.get("address") or ""
    raw_country = row_dict.get("country") or ""

    name_norm = normalize_name(raw_name)
    address_info = normalize_address(raw_address)
    country_norm = normalize_country(raw_country)

    return {
        "entity_id": str(entity_id),
        "name_norm": name_norm,
        "address_norm": address_info["address_norm"],
        "street": address_info["street"],
        "city": address_info["city"],
        "state": address_info["state"],
        "postal_code": address_info["postal_code"],
        "country": country_norm,
    }


def clean_text(text: Optional[str]) -> str:
    """Clean and normalize raw text string."""
    return normalize_name(text)


def normalize_entity_dataframe(
    df: pd.DataFrame,
    text_cols: Optional[list] = None
) -> pd.DataFrame:
    """Apply text normalization and standardization across DataFrame.

    Optimized for multi-million row DataFrames using unique value caching.

    Args:
        df: Input raw DataFrame.
        text_cols: Optional list of column names to clean.

    Returns:
        pd.DataFrame: Copy of DataFrame with normalized text columns.
    """
    df_out = df.copy()

    # Extract ID column defensively
    id_col = None
    for c in ["entity_id", "source1_entity_id", "id"]:
        if c in df_out.columns:
            id_col = c
            break

    if id_col is not None and id_col != "entity_id":
        df_out["entity_id"] = df_out[id_col].astype(str)
    elif id_col is None:
        df_out["entity_id"] = df_out.index.astype(str)
    else:
        df_out["entity_id"] = df_out["entity_id"].astype(str)

    # Name normalization with unique caching
    name_col = "business_name" if "business_name" in df_out.columns else ("name" if "name" in df_out.columns else None)
    if name_col is not None:
        unique_names = df_out[name_col].fillna("").unique()
        name_map = {name: normalize_name(name) for name in unique_names}
        df_out["name_norm"] = df_out[name_col].fillna("").map(name_map).fillna("")
    else:
        df_out["name_norm"] = ""

    # Address normalization with unique caching
    addr_col = "business_address" if "business_address" in df_out.columns else ("address" if "address" in df_out.columns else None)
    if addr_col is not None:
        unique_addrs = df_out[addr_col].fillna("").unique()
        addr_map = {addr: normalize_address(addr) for addr in unique_addrs}

        addr_parsed = df_out[addr_col].fillna("").map(addr_map)
        parsed_df = pd.DataFrame(addr_parsed.tolist(), index=df_out.index)

        df_out["street"] = parsed_df["street"]
        df_out["city"] = parsed_df["city"]
        df_out["state"] = parsed_df["state"]
        df_out["postal_code"] = parsed_df["postal_code"]
        df_out["address_norm"] = parsed_df["address_norm"]
    else:
        for col in ["street", "city", "state", "postal_code", "address_norm"]:
            df_out[col] = ""

    # Country normalization with unique caching
    if "country" in df_out.columns:
        unique_countries = df_out["country"].fillna("").unique()
        country_map = {c: normalize_country(c) for c in unique_countries}
        df_out["country"] = df_out["country"].fillna("").map(country_map).fillna("")
    else:
        df_out["country"] = ""

    return df_out



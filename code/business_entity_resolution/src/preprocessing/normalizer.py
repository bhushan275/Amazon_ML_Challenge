"""
Stage 1: Preprocessing & Normalization Module
Handles unicode normalization, legal suffix normalization, DBA handling,
address tokenization, street abbreviation expansion, landmark protection,
and open-set country string cleaning.
"""

import re
import unicodedata

# Dictionary of legal entity suffixes mapped to standard canonical tokens
LEGAL_SUFFIX_MAP = {
    r'\b(corporation|corp|corp\.)\b': 'corp',
    r'\b(incorporated|inc|inc\.)\b': 'inc',
    r'\b(private limited|pvt ltd|pvt\. ltd\.|pvt limited)\b': 'pvt_ltd',
    r'\b(limited|ltd|ltd\.)\b': 'ltd',
    r'\b(limited liability company|llc|l\.l\.c\.)\b': 'llc',
    r'\b(company|co|co\.)\b': 'co',
    r'\b(societe anonyme|s\.a\.|sa)\b': 'sa',
    r'\b(sarl|s\.a\.r\.l\.)\b': 'sarl',
    r'\b(gmbh|g\.m\.b\.h\.)\b': 'gmbh',
}

# Street & Address abbreviation expansion dictionary
ADDRESS_ABBR_MAP = {
    r'\b(st|st\.)\b': 'street',
    r'\b(rd|rd\.)\b': 'road',
    r'\b(ave|ave\.)\b': 'avenue',
    r'\b(blvd|blvd\.)\b': 'boulevard',
    r'\b(dr|dr\.)\b': 'drive',
    r'\b(pkwy|pkwy\.)\b': 'parkway',
    r'\b(ln|ln\.)\b': 'lane',
    r'\b(ste|ste\.)\b': 'suite',
    r'\b(fl|fl\.)\b': 'floor',
    r'\b(bldg|bldg\.)\b': 'building',
    r'\b(apt|apt\.)\b': 'apartment',
    r'\b(opp|opp\.)\b': 'opposite',
    r'\b(nr|nr\.)\b': 'near',
}


def strip_accents(text: str) -> str:
    """Normalize unicode text and convert accents to ASCII."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if not unicodedata.combining(c))


def clean_name(name: str) -> str:
    """
    Cleans business name:
    - Normalizes unicode & case
    - Replaces '&' with 'and'
    - Standardizes legal suffixes
    - Strips non-alphanumeric noise except spaces
    """
    if not name or not isinstance(name, str):
        return ""

    text = strip_accents(name).lower()
    text = re.sub(r'\s*&\s*', ' and ', text)

    # Standardize legal suffixes
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        text = re.sub(pattern, f' {replacement} ', text)

    # Remove special characters except alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_address(address: str) -> str:
    """
    Cleans business address:
    - Normalizes unicode & case
    - Expands common address abbreviations
    - Preserves landmark phrases ('near', 'opposite')
    - Collapses whitespace
    """
    if not address or not isinstance(address, str):
        return ""

    text = strip_accents(address).lower()

    # Expand address abbreviations
    for pattern, replacement in ADDRESS_ABBR_MAP.items():
        text = re.sub(pattern, f' {replacement} ', text)

    # Clean non-alphanumeric except spaces and commas
    text = re.sub(r'[^a-z0-9\s,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_country(country: str) -> str:
    """
    Cleans country string without filtering or hardcoding set of countries.
    Preserves open-set country representation (US, India, France, etc.).
    """
    if not country or not isinstance(country, str):
        return "unknown"
    return country.strip().upper()


def preprocess_record(record: dict) -> dict:
    """Preprocesses a single business entity record."""
    name = record.get('business_name', '')
    addr = record.get('business_address', '')
    cty = record.get('country', '')

    c_name = clean_name(name)
    c_addr = clean_address(addr)

    return {
        'entity_id': record.get('entity_id', ''),
        'business_name': name,
        'business_address': addr,
        'name_clean': c_name,
        'address_clean': c_addr,
        'combined_clean': f"{c_name} {c_addr}".strip(),
        'country': clean_country(cty)
    }

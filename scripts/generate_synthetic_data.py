#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Amazon ML Challenge 2026 - Business Entity Resolution.
Generates realistic multi-source business records with ground truth, realistic noise,
and open-set country representation (US, India in Train; US, India, France in Test).
"""

import os
import random
import csv
from pathlib import Path

# Seed for reproducibility
random.seed(42)

TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")

LEGAL_SUFFIXES = [
    ("Corporation", ["Corp", "Corp.", "Inc", "Corporation"]),
    ("Private Limited", ["Pvt Ltd", "Pvt. Ltd.", "Private Limited", "Pvt Limited"]),
    ("Limited", ["Ltd", "Ltd.", "Limited"]),
    ("Incorporated", ["Inc", "Inc.", "Incorporated"]),
    ("LLC", ["LLC", "L.L.C.", "Limited Liability Co"]),
    ("Société Anonyme", ["SA", "S.A.", "Société Anonyme"]),
]

STREET_TYPES = [
    ("Street", ["St", "St.", "Street"]),
    ("Road", ["Rd", "Rd.", "Road"]),
    ("Avenue", ["Ave", "Ave.", "Avenue"]),
    ("Boulevard", ["Blvd", "Blvd.", "Boulevard"]),
    ("Highway", ["Hwy", "Highway"]),
    ("Rue", ["R.", "Rue"]),
]

LANDMARKS = [
    "Near City Center", "Opposite Central Station", "Near SBI ATM",
    "Behind Grand Mall", "Adj. Metro Station", "Near Eiffel Tower",
    "Opp. Public Library", "Above HDFC Bank"
]

BASE_BUSINESSES = [
    # US Businesses
    ("Apex Global Logistics", "100 Industrial Parkway, Suite 400, Chicago, IL 60601", "US"),
    ("Quantum Tech Solutions", "550 Technology Square, Boston, MA 02139", "US"),
    ("Summit Federal Credit Union", "1200 Market Street, Philadelphia, PA 19107", "US"),
    ("Blue River Organics", "742 Evergreen Terrace, Springfield, OR 97477", "US"),
    ("Horizon Health Systems", "3300 Oakmont Boulevard, Austin, TX 78703", "US"),
    ("Vanguard Data Services", "1 Broadway, Floor 12, New York, NY 10004", "US"),
    ("Pacific Coast Coffee Roasters", "888 Pine Street, Seattle, WA 98101", "US"),
    ("Pinnacle Construction Group", "450 Peachtree St NW, Atlanta, GA 30308", "US"),
    ("Titan Auto Repair", "2100 Main Street, Houston, TX 77002", "US"),
    ("Sunrise Bakery & Cafe", "505 Elm Street, Denver, CO 80202", "US"),

    # India Businesses
    ("Tata Consultancy Services", "IT Park, Plot 22, Phase 3, Hinjewadi, Pune 411057", "India"),
    ("Infosys Digital Labs", "Electronic City, Hosur Road, Bangalore 560100", "India"),
    ("Reliance Retail Ventures", "Maker Chambers IV, Nariman Point, Mumbai 400021", "India"),
    ("HDFC Bank Regional Hub", "10 Kasturba Road, MG Road Corner, Bangalore 560001", "India"),
    ("Sri Krishna Sweets & Snacks", "45 Usman Road, T. Nagar, Chennai 600017", "India"),
    ("Sharma Traders & Wholesalers", "102 Chandni Chowk Road, New Delhi 110006", "India"),
    ("Greenfield Bio Tech", "Genome Valley, Shamirpet, Hyderabad 500078", "India"),
    ("Apollo Healthcare Clinic", "15 Greams Lane, Thousand Lights, Chennai 600006", "India"),
    ("Mahindra Logistics Center", "Gat No 120, Chakan Industrial Area, Pune 410501", "India"),
    ("Royal Enfield Auto Zone", "28 Mount Road, Guindy, Chennai 600032", "India"),

    # France Businesses (Unseen in train, present in test)
    ("Lumière Cinema & Media", "15 Boulevard des Capucines, 75002 Paris", "France"),
    ("AeroTech Industries France", "31 Rue de la République, 31000 Toulouse", "France"),
    ("Bistro de la Rose", "8 Place Bellecour, 69002 Lyon", "France"),
    ("Pharmacie Centrale Lyon", "42 Rue Victor Hugo, 69002 Lyon", "France"),
    ("Grand Vin de Bordeaux", "12 Cours du Chapeau Rouge, 33000 Bordeaux", "France"),
    ("TechnoPole Marseille", "50 Avenue du Prado, 13006 Marseille", "France"),
]


def introduce_noise_name(name):
    """Applies realistic name noise (abbreviations, punctuation, typos)."""
    words = name.split()
    # Punctuation & case
    if random.random() < 0.3:
        name = name.replace(" & ", " and ")
    elif random.random() < 0.3:
        name = name.replace(" and ", " & ")

    # Suffix variation
    for norm_suf, variants in LEGAL_SUFFIXES:
        if norm_suf.lower() in name.lower():
            replacement = random.choice(variants)
            name = name.replace(norm_suf, replacement)
            break

    # Typo / transposition
    if random.random() < 0.25 and len(name) > 8:
        idx = random.randint(3, len(name) - 3)
        name = name[:idx] + name[idx+1] + name[idx] + name[idx+2:]

    return name.strip()


def introduce_noise_address(address):
    """Applies realistic address noise (street abbreviation, landmark insertion, PIN drop)."""
    # Abbreviation
    for norm_st, variants in STREET_TYPES:
        if norm_st.lower() in address.lower():
            replacement = random.choice(variants)
            address = address.replace(norm_st, replacement)
            break

    # Landmark insertion
    if random.random() < 0.35:
        landmark = random.choice(LANDMARKS)
        address = f"{address}, {landmark}"

    # Omit PIN/Zip code (drop last 5-6 digit block)
    if random.random() < 0.3:
        tokens = address.split()
        tokens = [t for t in tokens if not (t.isdigit() and len(t) >= 5)]
        address = " ".join(tokens)

    return address.strip()


def generate_dataset():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic datasets...")

    # Filter base records for train (US, India only) vs test (US, India, France)
    train_base = [b for b in BASE_BUSINESSES if b[2] in ["US", "India"]]
    test_base = BASE_BUSINESSES  # Includes France

    # Generate Train Set
    train_s1, train_s2, train_s3, train_gt = create_source_files(train_base, prefix="TR", count=150)
    save_tsv(TRAIN_DIR / "train_source1.tsv", train_s1, ["entity_id", "business_name", "business_address", "country"])
    save_tsv(TRAIN_DIR / "train_source2.tsv", train_s2, ["entity_id", "business_name", "business_address", "country"])
    save_tsv(TRAIN_DIR / "train_source3.tsv", train_s3, ["entity_id", "business_name", "business_address", "country"])
    save_tsv(TRAIN_DIR / "train_ground_truth.tsv", train_gt, ["source1_entity_id", "matched_entity_ids"])

    # Generate Test Set
    test_s1, test_s2, test_s3, _ = create_source_files(test_base, prefix="TE", count=100)
    save_tsv(TEST_DIR / "test_source1.tsv", test_s1, ["entity_id", "business_name", "business_address", "country"])
    save_tsv(TEST_DIR / "test_source2.tsv", test_s2, ["entity_id", "business_name", "business_address", "country"])
    save_tsv(TEST_DIR / "test_source3.tsv", test_s3, ["entity_id", "business_name", "business_address", "country"])

    print(f"✅ Generated {len(train_s1)} Train S1 records ({len(train_s2)} S2, {len(train_s3)} S3)")
    print(f"✅ Generated {len(test_s1)} Test S1 records ({len(test_s2)} S2, {len(test_s3)} S3)")


def create_source_files(base_records, prefix, count):
    s1_rows = []
    s2_rows = []
    s3_rows = []
    gt_rows = []

    s2_counter = 1
    s3_counter = 1

    for i in range(1, count + 1):
        s1_id = f"S1-{prefix}-{i:05d}"
        base_name, base_addr, country = random.choice(base_records)

        # 1. Clean Reference Record in S1
        s1_rows.append({
            "entity_id": s1_id,
            "business_name": base_name,
            "business_address": base_addr,
            "country": country
        })

        matched_ids = []
        # Match probability: 30% singleton, 40% 1 match (S2 or S3), 30% multiple matches (S2 and S3)
        prob = random.random()
        if prob > 0.3:
            # Match in Source 2
            if random.random() < 0.7:
                s2_id = f"S2-{prefix}-{s2_counter:05d}"
                s2_counter += 1
                noisy_name = introduce_noise_name(base_name)
                noisy_addr = introduce_noise_address(base_addr)
                s2_rows.append({
                    "entity_id": s2_id,
                    "business_name": noisy_name,
                    "business_address": noisy_addr,
                    "country": country
                })
                matched_ids.append(s2_id)

            # Match in Source 3
            if random.random() < 0.6 or not matched_ids:
                s3_id = f"S3-{prefix}-{s3_counter:05d}"
                s3_counter += 1
                noisy_name = introduce_noise_name(base_name)
                noisy_addr = introduce_noise_address(base_addr)
                s3_rows.append({
                    "entity_id": s3_id,
                    "business_name": noisy_name,
                    "business_address": noisy_addr,
                    "country": country
                })
                matched_ids.append(s3_id)

        gt_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids)
        })

    # Add un-matched distractor entities to S2 and S3
    for _ in range(count // 2):
        b_name, b_addr, cty = random.choice(base_records)
        s2_id = f"S2-{prefix}-{s2_counter:05d}"
        s2_counter += 1
        s2_rows.append({
            "entity_id": s2_id,
            "business_name": introduce_noise_name(b_name + " Branch"),
            "business_address": introduce_noise_address(b_addr),
            "country": cty
        })

        b_name2, b_addr2, cty2 = random.choice(base_records)
        s3_id = f"S3-{prefix}-{s3_counter:05d}"
        s3_counter += 1
        s3_rows.append({
            "entity_id": s3_id,
            "business_name": introduce_noise_name(b_name2 + " Subsidiary"),
            "business_address": introduce_noise_address(b_addr2),
            "country": cty2
        })

    return s1_rows, s2_rows, s3_rows, gt_rows


def save_tsv(filepath, rows, fieldnames):
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    generate_dataset()

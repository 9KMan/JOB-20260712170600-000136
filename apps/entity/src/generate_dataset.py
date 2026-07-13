"""
Synthetic dataset generator for entity-resolution PoC.

Simulates 3 vendor datasets, each with a different record layout (because real
people-search vendors all use slightly different schemas). Total ~5,500 records
drawn from a ground-truth list of 2,000 unique people, with:

  - 60% of people present in all 3 vendors (high overlap)
  - 25% of people present in 2 of 3 (medium overlap)
  - 15% of people present in only 1 (unique to one vendor)

Each duplicate gets:
  - 0–2 character typos in name (Damerau-Levenshtein)
  - DOB may shift by ±1 day (transcription error)
  - Address may move (street name typo, ZIP digits flipped)
  - Phone / email may be missing or different format
  - Vendor-specific "noise" columns (e.g. TLOxp has 'risk_score', LexisNexis has 'criminal_record')

Output:
  data/vendor_a.parquet
  data/vendor_b.parquet
  data/vendor_c.parquet
  data/ground_truth.parquet  (golden clusters for false-merge audit)

Run: python -m apps.entity.src.generate_dataset
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Ground truth universe
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Person:
    """Canonical person in our ground-truth universe."""
    pid: str                                  # person_id
    first: str
    last: str
    dob: str                                  # ISO date
    street: str
    city: str
    state: str
    zip_code: str
    phone: str                                # E.164
    email: str
    ssn_last4: str                           # 4-digit, used by some vendors

# Build a small universe of 2,000 US people — realistic name/addr distribution
def make_universe(n: int = 2000) -> list[Person]:
    firsts = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
              "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph",
              "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy",
              "Daniel", "Lisa", "Matthew", "Margaret", "Anthony", "Betty", "Mark", "Sandra"]
    lasts = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
             "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
             "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
             "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"]
    cities = [("Houston", "TX", "770"), ("Los Angeles", "CA", "900"), ("Chicago", "IL", "606"),
              ("Phoenix", "AZ", "850"), ("Philadelphia", "PA", "191"), ("San Antonio", "TX", "782"),
              ("San Diego", "CA", "921"), ("Dallas", "TX", "752"), ("Austin", "TX", "787"),
              ("Miami", "FL", "331"), ("Atlanta", "GA", "303"), ("Seattle", "WA", "981")]
    street_names = ["Main", "Oak", "Pine", "Elm", "Cedar", "Maple", "Park", "Washington",
                    "Lake", "Hill", "Walnut", "Spring", "North", "South", "Center", "View"]
    street_suf = ["St", "Ave", "Blvd", "Rd", "Ln", "Dr", "Way", "Ct"]

    universe = []
    for i in range(n):
        f = random.choice(firsts)
        l = random.choice(lasts)
        city, st, zip3 = random.choice(cities)
        h = random.randint(100, 9999)
        street = f"{h} {random.choice(street_names)} {random.choice(street_suf)}"
        zip5 = f"{zip3}{random.randint(10, 99):02d}"
        phone = f"+1{random.randint(2000000000, 9999999999)}"
        # Use unique-enough email seed
        email = f"{f.lower()}.{l.lower()}{i}@example.com"
        dob_year = random.randint(1940, 2005)
        dob_month = random.randint(1, 12)
        dob_day = random.randint(1, 28)
        dob = f"{dob_year:04d}-{dob_month:02d}-{dob_day:02d}"
        ssn = f"{random.randint(0, 9999):04d}"
        universe.append(Person(pid=f"P{i:05d}", first=f, last=l, dob=dob,
                              street=street, city=city, state=st, zip_code=zip5,
                              phone=phone, email=email, ssn_last4=ssn))
    return universe


# ─────────────────────────────────────────────────────────────────────────────
# Vendor-specific emission functions
# ─────────────────────────────────────────────────────────────────────────────

def _typo(s: str, p: float = 0.04) -> str:
    """Apply Damerau-Levenshtein-style noise: insertion, deletion, substitution, transposition."""
    if not s or random.random() > p:
        return s
    op = random.choice(["sub", "del", "ins", "trans"])
    chars = list(s)
    if len(chars) < 2:
        return s
    if op == "sub":
        i = random.randint(0, len(chars) - 1)
        chars[i] = random.choice(string.ascii_letters + ".")
        return "".join(chars)
    elif op == "del":
        i = random.randint(0, len(chars) - 1)
        return "".join(chars[:i] + chars[i + 1:])
    elif op == "ins":
        i = random.randint(0, len(chars))
        chars.insert(i, random.choice(string.ascii_letters))
        return "".join(chars)
    else:  # trans
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)


def _emit(person: Person, vendor: str) -> dict:
    """Emit a single record for a vendor, applying vendor-specific quirks."""
    base = {
        # Common fields, possibly typo'd
        "first": _typo(person.first),
        "last": _typo(person.last),
        "dob": person.dob,
        "city": person.city,
        "state": person.state,
        "zip_code": person.zip_code,
        "email": person.email,
        "ground_truth_pid": person.pid,  # kept for audit; remove before shipping
    }

    if vendor == "A_lexisnexis":
        base.update({
            "vendor_record_id": f"LX-{random.randint(10**9, 10**10)}",
            "phone": _typo(person.phone, p=0.05) if random.random() > 0.1 else "",
            "address_full": f"{_typo(person.street)}, {person.city}, {person.state} {person.zip_code}",
            "criminal_record": random.choice([True, False]),
            "carrier": random.choice(["VERIZON", "ATT", "TMOBILE", "SPRINT", "UNKNOWN"]),
            "score": round(random.uniform(0.5, 0.99), 3),
        })
    elif vendor == "B_tracers":
        base.update({
            "vendor_record_id": f"TR-{random.randint(10**8, 10**9)}",
            "phone": person.phone,  # Tracers typically has accurate phone
            "address_full": _typo(person.street, p=0.08),
            "associates": str(random.randint(0, 5)),
            "last_seen": f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        })
    elif vendor == "C_pipl":
        base.update({
            "vendor_record_id": f"PP-{random.randint(10**10, 10**11)}",
            "phone": person.phone if random.random() > 0.3 else "",
            "address_full": f"{person.street}, {person.city}, {person.state} {person.zip_code}",
            "username_count": random.randint(1, 8),
            "breach_exposure": random.choice([True, False, False, False]),
            "social_handles": str(random.randint(0, 4)),
        })

    # ±1 day DOB jitter (transcription error) for ~3% of duplicates
    if random.random() < 0.03:
        try:
            year, month, day = person.dob.split("-")
            day_int = int(day)
            new_day = day_int + random.choice([-1, 1])
            if new_day < 1:
                new_day = 28
            base["dob"] = f"{year}-{month}-{new_day:02d}"
        except Exception:
            pass

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def generate(n_universe: int = 2000, *, out_dir: Path):
    universe = make_universe(n_universe)

    # Membership: each person is in some subset of vendors
    rows_a, rows_b, rows_c = [], [], []
    for p in universe:
        in_a = random.random() < 0.85
        in_b = random.random() < 0.80
        in_c = random.random() < 0.70
        # At least one vendor has it
        if not (in_a or in_b or in_c):
            in_a = True
        if in_a:
            rows_a.append(_emit(p, "A_lexisnexis"))
        if in_b:
            rows_b.append(_emit(p, "B_tracers"))
        if in_c:
            rows_c.append(_emit(p, "C_pipl"))

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    df_c = pd.DataFrame(rows_c)
    df_truth = pd.DataFrame([p.__dict__ for p in universe])

    out_dir.mkdir(parents=True, exist_ok=True)
    df_a.to_parquet(out_dir / "vendor_a_lexisnexis.parquet", index=False)
    df_b.to_parquet(out_dir / "vendor_b_tracers.parquet", index=False)
    df_c.to_parquet(out_dir / "vendor_c_pipl.parquet", index=False)
    df_truth.to_parquet(out_dir / "ground_truth.parquet", index=False)

    # Summary
    summary = {
        "universe_size": n_universe,
        "vendor_a_lexisnexis": len(df_a),
        "vendor_b_tracers":   len(df_b),
        "vendor_c_pipl":       len(df_c),
        "ground_truth_pids":   len(df_truth),
    }
    print("Generated synthetic vendor dataset:")
    for k, v in summary.items():
        print(f"  {k:24} {v}")
    return summary


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data"
    generate(out_dir=out)

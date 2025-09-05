# sales_assistant.py
# ------------------------------------------
# Core backend for the Kitchenware Sales Assistant
# Robust location matching + batching
# Requires: pandas, groq
# ------------------------------------------

from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
import pandas as pd
import unicodedata
import re
import difflib

try:
    from groq import Groq
except ImportError as e:
    raise ImportError("groq package not found. Install with: pip install groq") from e


DEFAULT_MODEL = "gemma2-9b-it"


def get_groq_client(api_key: Optional[str] = None) -> Groq:
    """Return an authenticated Groq client, reading the key from arg or env."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("Missing Groq API key. Set GROQ_API_KEY env var or pass api_key argument.")
    return Groq(api_key=key)


def _resolve_location_column(df: pd.DataFrame) -> Optional[str]:
    """Find which column to use as 'Location' (flexible names)."""
    candidates = ["Location", "location", "city", "City", "region", "Region", "town", "Town"]
    # check direct presence first
    for cand in candidates:
        if cand in df.columns:
            return cand
    # case-insensitive fallback
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def normalize_string(s: Optional[str]) -> str:
    """Normalize a string: remove accents, punctuation, extra spaces, lowercase."""
    if s is None:
        return ""
    s = str(s)
    # Unicode normalize and drop combining diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Lowercase
    s = s.lower()
    # Replace non-alphanumeric with space
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def matches_location(value: Optional[str], target_norm: str, threshold: float = 0.80) -> bool:
    """
    Return True if 'value' matches 'target_norm' using a sequence:
      - exact normalized match
      - token (word) membership
      - substring inclusion
      - fuzzy similarity (difflib ratio)
    This covers entries like "mumbai", "mumbai, maharashtra", "mumbai suburb", "mumbai - andheri", etc.
    """
    if not target_norm:
        return False
    val_norm = normalize_string(value)
    if not val_norm:
        return False
    if val_norm == target_norm:
        return True
    # token match (target as a whole word)
    val_tokens = val_norm.split()
    if target_norm in val_tokens:
        return True
    # substring (target appears somewhere)
    if target_norm in val_norm:
        return True
    # fuzzy similarity
    ratio = difflib.SequenceMatcher(None, val_norm, target_norm).ratio()
    if ratio >= threshold:
        return True
    # token-set overlap score (optional additional safe check)
    target_tokens = set(target_norm.split())
    if target_tokens and val_tokens:
        inter = target_tokens.intersection(set(val_tokens))
        jaccard = len(inter) / max(1, len(target_tokens.union(set(val_tokens))))
        if jaccard >= 0.6:
            return True
    return False


def build_prompt(customer_name: str, customer_location: str, business_category: str, freq_purchase: str) -> str:
    """Construct the mini prompt for a single customer (used in batching)."""
    return f"- Name: {customer_name}, Location: {customer_location}, Frequent Category: {freq_purchase}"


def generate_messages_for_df(
    df: pd.DataFrame,
    business_category: str,
    default_location: str,
    client: Groq,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 300,
    batch_size: int = 5,
) -> pd.DataFrame:
    """
    Generate promotional messages for each row in df, using batch requests.
    Strict: only customers matching default_location (robust matching) are included.
    """

    # find the column that contains the location/city
    location_col = _resolve_location_column(df)
    if not location_col:
        return pd.DataFrame(columns=[
            "Name", "Email", "Location", "Category",
            "Frequently_Purchased_Category", "Message"
        ])

    # normalize the target location once
    target_norm = normalize_string(default_location)

    # robust filter (apply matches_location)
    mask = df[location_col].astype(str).apply(lambda v: matches_location(v, target_norm))
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        return pd.DataFrame(columns=[
            "Name", "Email", "Location", "Category",
            "Frequently_Purchased_Category", "Message"
        ])

    rows: List[Dict[str, Any]] = []

    # fallback mapping for columns
    fallback_map = {
        "Name": ["Name", "name", "customer_name"],
        "Email": ["Email", "email", "email_id"],
        "Location": [location_col, "Location", "location", "city"],
        "Frequently_Purchased_Category": ["Frequently_Purchased_Category", "category", "segment", "fav_category"],
    }

    def pick_from_row(r: Dict[str, Any], keys: List[str]) -> Optional[Any]:
        for k in keys:
            if k in r and pd.notna(r[k]):
                return r[k]
        return None

    # Process in batches
    for start in range(0, len(df_filtered), batch_size):
        batch = df_filtered.iloc[start:start + batch_size]

        # Build batch prompt
        prompt_lines = [
            f"You are a Sales Assistant for a business dealing in {business_category}.",
            "Write a short (4–5 sentence) promotional message for each customer below, referencing their city and frequent category.",
        ]
        for i, (_, row) in enumerate(batch.iterrows(), start=1):
            r = row.to_dict()
            customer_name = pick_from_row(r, fallback_map["Name"]) or "Customer"
            customer_location = pick_from_row(r, fallback_map["Location"]) or default_location
            freq_purchase = pick_from_row(r, fallback_map["Frequently_Purchased_Category"]) or "General"
            prompt_lines.append(f"\nCustomer {i}: {build_prompt(customer_name, customer_location, business_category, freq_purchase)}")

        prompt_lines.append(
            "\n⚠️ Output strictly in this format (one line per customer):\n"
            "Customer 1: <promotional message>\n"
            "Customer 2: <promotional message>\n"
        )
        prompt = "\n".join(prompt_lines)

        # Single API call for the batch
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful sales assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        output_text = resp.choices[0].message.content.strip()

        # Parse lines starting with "Customer X:"
        out_lines = [ln.strip() for ln in output_text.splitlines() if ln.strip()]
        for out_line in out_lines:
            if out_line.lower().startswith("customer"):
                try:
                    cust_id, msg = out_line.split(":", 1)
                    cust_idx = int(cust_id.strip().split()[1]) - 1
                    if 0 <= cust_idx < len(batch):
                        r = batch.iloc[cust_idx].to_dict()
                        rows.append({
                            "Name": pick_from_row(r, fallback_map["Name"]) or "Customer",
                            "Email": pick_from_row(r, fallback_map["Email"]) or "N/A",
                            "Location": pick_from_row(r, fallback_map["Location"]) or default_location,
                            "Category": business_category,
                            "Frequently_Purchased_Category": pick_from_row(r, fallback_map["Frequently_Purchased_Category"]) or "General",
                            "Message": msg.strip(),
                        })
                except Exception:
                    # ignore parse errors and continue
                    continue

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sales messages from a CSV.")
    parser.add_argument("--input", required=True, help="Path to input CSV/Excel")
    parser.add_argument("--output", default="customised_messages.csv", help="Path to output CSV")
    parser.add_argument("--category", required=True, help="Business category, e.g., Kitchenware")
    parser.add_argument("--location", required=True, help="Default/Target location")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq chat model name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--api_key", default=None, help="Groq API key (overrides env)")
    args = parser.parse_args()

    # Load input
    if args.input.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    client = get_groq_client(args.api_key)
    out_df = generate_messages_for_df(
        df,
        business_category=args.category,
        default_location=args.location,
        client=client,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    out_df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

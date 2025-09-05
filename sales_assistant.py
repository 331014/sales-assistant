# sales_assistant.py
# ------------------------------------------
# Core backend for the Kitchenware Sales Assistant
# Robust location matching + batching
# Recommendations are based on Frequent Category,
# city is used only for tone/context.
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
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("Missing Groq API key. Set GROQ_API_KEY env var or pass api_key argument.")
    return Groq(api_key=key)


def _resolve_location_column(df: pd.DataFrame) -> Optional[str]:
    """Find which column to use as 'Location' (flexible names)."""
    candidates = ["Location", "location", "city", "City", "region", "Region", "town", "Town"]
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
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def matches_location(value: Optional[str], target_norm: str, threshold: float = 0.80) -> bool:
    """Robust matching: exact, token, substring, fuzzy similarity."""
    if not target_norm:
        return False
    val_norm = normalize_string(value)
    if not val_norm:
        return False
    if val_norm == target_norm:
        return True
    if target_norm in val_norm.split():
        return True
    if target_norm in val_norm:
        return True
    if difflib.SequenceMatcher(None, val_norm, target_norm).ratio() >= threshold:
        return True
    return False


def build_prompt(customer_name: str, customer_location: str, business_category: str, freq_purchase: str) -> str:
    """Construct input lines for each customer."""
    return (
        f"- Name: {customer_name}\n"
        f"- City: {customer_location}\n"
        f"- Frequently Purchased Category: {freq_purchase}"
    )


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
    """Generate promotional messages based on Frequent Category, filtered by city."""

    location_col = _resolve_location_column(df)
    if not location_col:
        return pd.DataFrame(columns=[
            "Name", "Email", "Location", "Category",
            "Frequently_Purchased_Category", "Message"
        ])

    target_norm = normalize_string(default_location)
    mask = df[location_col].astype(str).apply(lambda v: matches_location(v, target_norm))
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        return pd.DataFrame(columns=[
            "Name", "Email", "Location", "Category",
            "Frequently_Purchased_Category", "Message"
        ])

    rows: List[Dict[str, Any]] = []

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

    for start in range(0, len(df_filtered), batch_size):
        batch = df_filtered.iloc[start:start + batch_size]

        prompt_lines = [
            f"You are a Sales Assistant for a business dealing in {business_category}.",
            "Write a short (4–5 sentence) promotional message for each customer below.",
            "⚠️ Important: Recommend products primarily based on their 'Frequently Purchased Category'.",
            "Use the city only to make the message relatable in tone or greeting, not for deciding products.",
        ]
        for i, (_, row) in enumerate(batch.iterrows(), start=1):
            r = row.to_dict()
            customer_name = pick_from_row(r, fallback_map["Name"]) or "Customer"
            customer_location = pick_from_row(r, fallback_map["Location"]) or default_location
            freq_purchase = pick_from_row(r, fallback_map["Frequently_Purchased_Category"]) or "General"
            prompt_lines.append(f"\nCustomer {i}:\n{build_prompt(customer_name, customer_location, business_category, freq_purchase)}")

        prompt_lines.append(
            "\n⚠️ Output strictly in this format:\n"
            "Customer 1: <promotional message>\n"
            "Customer 2: <promotional message>\n"
        )
        prompt = "\n".join(prompt_lines)

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

        for line in output_text.splitlines():
            if not line.strip():
                continue
            if line.lower().startswith("customer"):
                try:
                    cust_id, msg = line.split(":", 1)
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
                    continue

    return pd.DataFrame(rows)

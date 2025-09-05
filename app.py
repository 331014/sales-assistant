# app.py
# ------------------------------------------
# Streamlit frontend for the Kitchenware Sales Assistant
# Run with: streamlit run app.py
# ------------------------------------------

import pandas as pd
import streamlit as st

from sales_assistant import (
    get_groq_client,
    generate_messages_for_df,
    DEFAULT_MODEL,
    _resolve_location_column,
    normalize_string,
    matches_location,
)

st.set_page_config(page_title="Kitchenware Sales Assistant", page_icon="🍳", layout="wide")

st.title("🍳 Kitchenware Sales Assistant")
st.write("Upload your leads file, set your options, and generate personalized outreach messages.")

with st.sidebar:
    st.header("🔐 API & Model")
    api_key_input = st.text_input("Groq API Key (optional)", type="password")

    # small curated list + manual option
    available_models = [
        "gemma2-9b-it",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b",
    ]
    model_choice = st.selectbox("Choose model", options=available_models + ["Other (enter manually)"], index=0)
    if model_choice == "Other (enter manually)":
        model_selected = st.text_input("Enter model name", value=DEFAULT_MODEL)
    else:
        model_selected = model_choice

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
    max_tokens = st.number_input("Max tokens", min_value=50, max_value=1000, value=200, step=50)
    batch_size = st.number_input("Batch size", min_value=1, max_value=50, value=5, step=1)

    st.header("⚙️ Campaign Settings")
    business_category = st.text_input("Business Category", value="Kitchenware")

uploaded = st.file_uploader("Upload leads file (.csv or .xlsx)", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.subheader("👀 Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # detect location column (flexible)
    location_col = _resolve_location_column(df)
    if not location_col:
        st.error("No column found for location (expected 'Location' or 'city' or similar). Please rename the column and reupload.")
        st.stop()

    # build a list of unique city display names (preserve original casing for display)
    unique_raw = df[location_col].dropna().astype(str).str.strip().tolist()
    # preserve order & uniqueness (first occurrence keeps original casing)
    seen = set()
    unique_display = []
    for v in unique_raw:
        key = normalize_string(v)
        if key not in seen:
            seen.add(key)
            unique_display.append(v.strip())

    # Add an option to enter manually
    options = unique_display + ["Other (enter manually)"]
    choice = st.selectbox("Default/Target Location", options=options, index=0, format_func=lambda x: x)

    if choice == "Other (enter manually)":
        manual_city = st.text_input("Enter city manually (e.g., Mumbai)", value="")
        chosen_display = manual_city.strip()
    else:
        chosen_display = choice

    if not chosen_display:
        st.info("Choose or enter a target location to enable generation.")
    else:
        if st.button("🚀 Generate Messages", type="primary"):
            try:
                client = get_groq_client(api_key_input or None)
            except Exception as e:
                st.error(f"API key error: {e}")
                st.stop()

            # robust filter in the frontend using the same matching logic
            chosen_norm = normalize_string(chosen_display)
            mask = df[location_col].astype(str).apply(lambda v: matches_location(v, chosen_norm))
            filtered_df = df[mask].copy()

            if filtered_df.empty:
                st.warning(f"No leads found for location: {chosen_display} (after robust matching).")
            else:
                with st.spinner("Generating messages..."):
                    try:
                        out_df = generate_messages_for_df(
                            df=filtered_df,
                            business_category=business_category.strip() or "Kitchenware",
                            default_location=chosen_display,
                            client=client,
                            model=model_selected.strip() or DEFAULT_MODEL,
                            temperature=float(temperature),
                            max_tokens=int(max_tokens),
                            batch_size=int(batch_size),
                        )
                    except Exception as e:
                        # if the error looks like an invalid/decommissioned model, try fallback once
                        err_str = str(e).lower()
                        if "decommission" in err_str or "model_decommissioned" in err_str or "invalid_request_error" in err_str:
                            st.warning(f"Selected model appears unavailable. Retrying with default model '{DEFAULT_MODEL}'...")
                            try:
                                out_df = generate_messages_for_df(
                                    df=filtered_df,
                                    business_category=business_category.strip() or "Kitchenware",
                                    default_location=chosen_display,
                                    client=client,
                                    model=DEFAULT_MODEL,
                                    temperature=float(temperature),
                                    max_tokens=int(max_tokens),
                                    batch_size=int(batch_size),
                                )
                                st.info(f"Retried using default model: {DEFAULT_MODEL}")
                            except Exception as e2:
                                st.error(f"Retry with default model failed: {e2}")
                                st.stop()
                        else:
                            st.error(f"Generation failed: {e}")
                            st.stop()

                if out_df.empty:
                    st.warning(f"No messages generated for {chosen_display}")
                else:
                    st.success("Done! 🎉")
                    st.subheader("📬 Generated Messages")
                    st.dataframe(out_df, use_container_width=True)

                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_bytes,
                        file_name="customised_messages.csv",
                        mime="text/csv",
                    )

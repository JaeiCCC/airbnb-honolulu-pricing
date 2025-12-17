#!/usr/bin/env python
# coding: utf-8

# # In this file, we do all the complicated data cleaning and feature engineering based on the raw data
# ## The whole process is very long and it includes many intermediate outputs for checking and run-on manipulations.
# ## The final dataset after cleaning and feature engineering is called "Listing_Honolulu.csv"

# In[ ]:


import pandas as pd


listings_df = pd.read_csv('listings.csv')


listings_2_df = pd.read_csv('listings_2.csv')


print("listings.csv columns:")
print(listings_df.columns.tolist())
print(f"\nTotal columns: {len(listings_df.columns)}")
print("\n" + "=" * 50 + "\n")


print("listings_2.csv columns:")
print(listings_2_df.columns.tolist())
print(f"\nTotal columns: {len(listings_2_df.columns)}")


# In[ ]:


# Select 24 columns from listings.csv
selected_columns_from_listings1 = [
    'number_of_reviews_l30d',
    'availability_30',
    'calculated_host_listings_count_entire_homes',
    'review_scores_rating',
    'review_scores_cleanliness',
    'review_scores_location',
    'review_scores_communication',
    'instant_bookable',
    'maximum_nights',
    'maximum_nights_avg_ntm',
    'minimum_nights_avg_ntm',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'amenities',
    'property_type',
    'host_response_time',
    'host_response_rate',
    'host_acceptance_rate',
    'host_is_superhost',
    'host_listings_count',
    'host_has_profile_pic',
    'host_since'
]

# Keep id plus the selected columns from listings.csv
columns_to_select = ['id'] + selected_columns_from_listings1
selected_data = listings_df[columns_to_select]

# Merge selected columns from listings.csv into listings_2.csv by id
listings_3_df = listings_2_df.merge(selected_data, on='id', how='left')

# Desired column order
column_order = [
    'id',
    'name',
    'property_type',
    'room_type',
    'accommodates',
    'beds',
    'bedrooms',
    'bathrooms',
    'host_id',
    'host_name',
    'host_since',
    'host_is_superhost',
    'host_listings_count',
    'calculated_host_listings_count',
    'calculated_host_listings_count_entire_homes',
    'host_response_time',
    'host_response_rate',
    'host_acceptance_rate',
    'host_has_profile_pic',
    'neighbourhood_group',
    'neighbourhood',
    'latitude',
    'longitude',
    'availability_365',
    'availability_30',
    'minimum_nights',
    'maximum_nights',
    'minimum_nights_avg_ntm',
    'maximum_nights_avg_ntm',
    'number_of_reviews',
    'number_of_reviews_ltm',
    'number_of_reviews_l30d',
    'reviews_per_month',
    'last_review',
    'review_scores_rating',
    'review_scores_cleanliness',
    'review_scores_location',
    'review_scores_communication',
    'instant_bookable',
    'amenities',
    'price',
    'license'
]

# Identify which columns are available after the merge
available_columns = [col for col in column_order if col in listings_3_df.columns]
missing_columns = [col for col in column_order if col not in listings_3_df.columns]

print("Available columns:", len(available_columns))
print("Missing columns:", missing_columns)

# Reorder columns using the desired order (only those that exist)
listings_3_df = listings_3_df[available_columns]

# Show final dataset columns and shape
print("\nlistings_3.csv columns and order:")
print(listings_3_df.columns.tolist())
print(f"\nTotal columns: {len(listings_3_df.columns)}")
print(f"Total rows: {len(listings_3_df)}")

# Save to listings_3.csv
listings_3_df.to_csv('listings_3.csv', index=False)
print("\nSaved listings_3.csv successfully")


# In[ ]:


listings_3_df = pd.read_csv('listings_3.csv')

variables_to_analyze = [
    'property_type',
    'room_type',
    'host_since',
    'host_response_time',
    'neighbourhood_group',
    'last_review',
    'amenities'
]

def sanitize_text(x: str) -> str:
    return str(x).strip().encode('utf-8', 'ignore').decode('utf-8', 'ignore')

print("=" * 80)
print("Variable analysis report (compact)")
print("=" * 80)

from collections import Counter
import ast
import json

top_k = 30  # show top 30 categories for standard categorical vars
top_k_amenities = 50  # show top 50 amenities

for var in variables_to_analyze:
    if var not in listings_3_df.columns:
        print(f"\n❌ Variable '{var}' not found in dataset")
        continue

    print(f"\n{'=' * 80}")
    print(f"Variable: {var}")
    print(f"{'=' * 80}")

    dtype = listings_3_df[var].dtype
    print(f"Data type: {dtype}")

    missing_count = listings_3_df[var].isna().sum()
    total_count = len(listings_3_df)
    missing_pct = (missing_count / total_count) * 100
    print(f"Missing values: {missing_count} / {total_count} ({missing_pct:.2f}%)")

    if dtype == 'object' or listings_3_df[var].dtype.name == 'object':
        if var == 'amenities':
            print("\n⚠️  Note: amenities contains multiple items; parsing and counting...")
            amenity_counter = Counter()
            for _, amenity_str in listings_3_df[var].dropna().items():
                amenity_str_clean = sanitize_text(amenity_str)
                parsed = None
                for parser in (ast.literal_eval, json.loads):
                    try:
                        parsed = parser(amenity_str_clean)
                        break
                    except Exception:
                        parsed = None
                if isinstance(parsed, list):
                    items = [sanitize_text(i) for i in parsed if sanitize_text(i)]
                else:
                    if amenity_str_clean.startswith('[') and amenity_str_clean.endswith(']'):
                        content = amenity_str_clean[1:-1]
                        items = [sanitize_text(x) for x in content.split('\",\"') if sanitize_text(x)]
                    else:
                        items = [amenity_str_clean] if amenity_str_clean else []
                amenity_counter.update(items)
            total_unique = len(amenity_counter)
            print(f"Total unique amenities: {total_unique}")
            print(f"Top {top_k_amenities} amenities:")
            for i, (amenity, cnt) in enumerate(amenity_counter.most_common(top_k_amenities), 1):
                print(f"  {i}. \"{amenity}\" ({cnt} occurrences)")
            if total_unique > top_k_amenities:
                print(f"... {total_unique - top_k_amenities} more not shown")
        else:
            values = listings_3_df[var].dropna().astype(str).map(sanitize_text)
            vc = values.value_counts()
            unique_count = len(vc)
            print(f"Unique categories: {unique_count}")
            print(f"Top {min(top_k, unique_count)} categories:")
            for i, (value, cnt) in enumerate(vc.head(top_k).items(), 1):
                print(f"  {i}. '{value}' ({cnt} occurrences)")
            if unique_count > top_k:
                print(f"... {unique_count - top_k} more not shown")
    else:
        print("\nNumeric summary:")
        print(listings_3_df[var].describe())

print(f"\n{'=' * 80}")
print("Analysis complete")
print(f"{'=' * 80}")


# In[ ]:


from pathlib import Path
from collections import Counter
import ast
import json

listings_3_df = pd.read_csv('listings_3.csv')

variables_to_analyze = [
    'property_type',
    'room_type',
    'host_since',
    'host_response_time',
    'neighbourhood_group',
    'last_review',
    'amenities'
]

def sanitize_text(x: str) -> str:
    return str(x).strip().encode('utf-8', 'ignore').decode('utf-8', 'ignore')

top_k_print = 30  # limit how many common categories are written to file
report_path = Path('listing3_analysis.txt')

with report_path.open('w', encoding='utf-8') as f:
    f.write('=' * 80 + '\n')
    f.write('Variable analysis report (full file to avoid notebook truncation)\n')
    f.write('=' * 80 + '\n\n')
    total_count = len(listings_3_df)

    for var in variables_to_analyze:
        f.write('\n' + '=' * 80 + '\n')
        f.write(f"Variable: {var}\n")
        f.write('=' * 80 + '\n')
        if var not in listings_3_df.columns:
            f.write("❌ Variable not found in dataset\n")
            continue
        dtype = listings_3_df[var].dtype
        missing_count = listings_3_df[var].isna().sum()
        missing_pct = (missing_count / total_count) * 100
        f.write(f"Data type: {dtype}\n")
        f.write(f"Missing values: {missing_count} / {total_count} ({missing_pct:.2f}%)\n")

        if dtype == 'object' or listings_3_df[var].dtype.name == 'object':
            if var == 'amenities':
                f.write('\namenities contains multiple values; parsing full frequency counts...\n')
                amenity_counter = Counter()
                for _, amenity_str in listings_3_df[var].dropna().items():
                    amenity_str_clean = sanitize_text(amenity_str)
                    parsed = None
                    for parser in (ast.literal_eval, json.loads):
                        try:
                            parsed = parser(amenity_str_clean)
                            break
                        except Exception:
                            parsed = None
                    if isinstance(parsed, list):
                        items = [sanitize_text(i) for i in parsed if sanitize_text(i)]
                    else:
                        if amenity_str_clean.startswith('[') and amenity_str_clean.endswith(']'):
                            content = amenity_str_clean[1:-1]
                            items = [sanitize_text(x) for x in content.split('\",\"') if sanitize_text(x)]
                        else:
                            items = [amenity_str_clean] if amenity_str_clean else []
                    amenity_counter.update(items)
                total_unique = len(amenity_counter)
                f.write(f"Total unique amenities: {total_unique}\n")
                f.write("All amenities with counts:\n")
                for amenity, cnt in amenity_counter.most_common():
                    f.write(f"  \"{amenity}\" ({cnt} occurrences)\n")
            else:
                values = listings_3_df[var].dropna().astype(str).map(sanitize_text)
                vc = values.value_counts()
                unique_count = len(vc)
                f.write(f"Unique categories: {unique_count}\n")
                f.write(f"Top {min(top_k_print, unique_count)} most common categories:\n")
                for value, cnt in vc.head(top_k_print).items():
                    f.write(f"  '{value}' ({cnt} occurrences)\n")
                if unique_count > top_k_print:
                    f.write(f"... {unique_count - top_k_print} more not shown; increase top_k_print to include all\n")
        else:
            f.write('\nNumeric summary:\n')
            f.write(str(listings_3_df[var].describe()) + '\n')

    f.write('\n' + '=' * 80 + '\n')
    f.write('Analysis complete\n')
    f.write('=' * 80 + '\n')

print(f"Analysis written to file: {report_path.resolve()}")



# In[ ]:





# In[ ]:





# In[ ]:


import re
import ast
import json

# Parse the amenities string into a normalized set of lowercased values
def parse_amenities_cell(s):
    if pd.isna(s):
        return set()
    raw = str(s)
    parsed = None
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(raw)
            break
        except Exception:
            parsed = None
    if isinstance(parsed, list):
        items = [str(x) for x in parsed]
    else:
        if raw.startswith('[') and raw.endswith(']'):
            content = raw[1:-1]
            items = [p.strip(' "\'\n\t') for p in content.split('","')]
        else:
            items = [raw]
    def norm(x: str) -> str:
        return re.sub(r"\s+", " ", sanitize_text(x).lower())
    return set(filter(None, (norm(x) for x in items)))

# Check whether any keyword is present in the amenity set
def has_keyword(items: set, keywords):
    kw_lower = [k.lower() for k in keywords]
    for amen in items:
        for kw in kw_lower:
            if kw in amen:
                return True
    return False

# Define 10 target amenities and related keyword variants
amenity_keyword_map = {
    'has_air_conditioning': [
        'air conditioning', 'a/c', 'ac', 'ac - split type ductless system',
        'window ac', 'window a/c', 'central air conditioning', 'portable air conditioning',
        'split ac', 'wall ac', 'ductless ac'
    ],
    'has_hot_water': ['hot water'],
    'has_kitchen': ['kitchen', 'kitchenette'],
    'has_gym': [
        'gym', 'fitness center', 'fitness room', 'fitness',
        'exercise equipment', 'private gym', 'shared gym'
    ],
    'has_wifi': ['wifi', 'wi-fi', 'wireless internet', 'ethernet connection'],
    'has_hair_dryer': ['hair dryer', 'hair-dryer', 'hairdryer'],
    'has_microwave': ['microwave'],
    'has_refrigerator': ['refrigerator', 'fridge', 'mini fridge', 'mini-fridge'],
    'has_pool': [
        'pool', 'swimming pool', 'private pool', 'shared pool', 'community pool',
        'lap pool', 'plunge pool', 'infinity pool', 'indoor pool', 'outdoor pool'
    ],
    'has_washer': [
        'washer', 'washing machine', 'laundry', 'free washer', 'paid washer',
        'laundry washer', 'laundry machine'
    ],
}

parsed_amenities_series = listings_3_df['amenities'].apply(parse_amenities_cell)

for col_name, keywords in amenity_keyword_map.items():
    listings_3_df[col_name] = parsed_amenities_series.apply(lambda items: has_keyword(items, keywords))

summary = {col: listings_3_df[col].value_counts(dropna=False).to_dict() for col in amenity_keyword_map}
print("New boolean columns (10):")
for col, counts in summary.items():
    true_cnt = counts.get(True, 0)
    false_cnt = counts.get(False, 0)
    print(f"  {col}: True={true_cnt}, False={false_cnt}")

output_path = 'listings_4.csv'
listings_3_df.to_csv(output_path, index=False)
print(f"Saved file with 10 boolean columns: {output_path}")



# In[ ]:


import pandas as pd

prop_df = pd.read_csv('listings_4.csv')

# Main property types (case-insensitive)
main_types = {
    'Entire condo': 'is_entire_condo',
    'Entire rental unit': 'is_entire_rental_unit',
    'Entire home': 'is_entire_home',
    'Room in hotel': 'is_room_in_hotel',
    'Entire serviced apartment': 'is_entire_serviced_apartment',
    'Entire townhouse': 'is_entire_townhouse'
}

def norm_prop(x: str) -> str:
    return sanitize_text(x).lower()

prop_lower = prop_df['property_type'].fillna('').map(norm_prop)

for label, col_name in main_types.items():
    prop_df[col_name] = prop_lower.eq(label.lower())

print("property_type boolean flags:")
for label, col_name in main_types.items():
    counts = prop_df[col_name].value_counts(dropna=False).to_dict()
    print(f"  {col_name} ({label}): True={counts.get(True,0)}, False={counts.get(False,0)}")

output_path = 'listings_5.csv'
prop_df.to_csv(output_path, index=False)
print(f"Saved file with property_type flags: {output_path}")


# In[ ]:


import pandas as pd

reorder_df = pd.read_csv('listings_5.csv')
flag_cols = [
    'is_entire_condo',
    'is_entire_rental_unit',
    'is_entire_home',
    'is_room_in_hotel',
    'is_entire_serviced_apartment',
    'is_entire_townhouse'
]

missing = [c for c in flag_cols if c not in reorder_df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

base_cols = [c for c in reorder_df.columns if c not in flag_cols]
if 'property_type' not in base_cols:
    raise ValueError("property_type not present; cannot reorder")

idx = base_cols.index('property_type')
new_order = base_cols[:idx + 1] + flag_cols + base_cols[idx + 1:]
reorder_df = reorder_df[new_order]

reorder_df.to_csv('listings_5.csv', index=False)

start = max(idx - 2, 0)
end = min(idx + 1 + len(flag_cols) + 2, len(new_order))
print("Column order around property_type:")
print(new_order[start:end])
print("Saved reordered file to listings_5.csv")



# In[ ]:


import pandas as pd

cols_to_drop = ['name', 'property_type', 'host_name', 'last_review', 'amenities', 'host_id', 'host_years.1']
df = pd.read_csv('listings_5.csv')

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
output_path = 'listings_6.csv'
df.to_csv(output_path, index=False)

print(f"Dropped columns {cols_to_drop} and saved to {output_path}")
print(f"Column count: {df.shape[1]}, rows: {df.shape[0]}")


# In[ ]:


import pandas as pd

src = 'listings_6.csv'
df = pd.read_csv(src)

honolulu_df = df[df['neighbourhood_group'] == 'Honolulu']
hawaii_df = df[df['neighbourhood_group'] == 'Hawaii']

honolulu_path = 'listing_honululu.csv'
hawaii_path = 'listing_hawaii.csv'

honolulu_df.to_csv(honolulu_path, index=False)
hawaii_df.to_csv(hawaii_path, index=False)

print(f"Honolulu rows: {len(honolulu_df)} saved to {honolulu_path}")
print(f"Hawaii rows: {len(hawaii_df)} saved to {hawaii_path}")


# In[ ]:


import pandas as pd
from pathlib import Path

hon_path = 'listing_honolulu.csv'
report_path = Path('listing_honolulu_report.txt')

df = pd.read_csv(hon_path)

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

with report_path.open('w', encoding='utf-8') as f:
    f.write(f"File: {hon_path}\n")
    f.write(f"Rows: {len(df)}, Columns: {df.shape[1]}\n\n")
    f.write("Column dtypes:\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nMissing value summary:\n")
    for col in df.columns:
        f.write(f"{col}: {missing[col]} missing ({missing_pct[col]}%)\n")

print("Report written to:", report_path.resolve())
print("(Open the file to view full output)")


# In[ ]:


import pandas as pd

df = pd.read_csv('listing_honululu.csv')

missing_idx = df.index[df['host_years'].isna()].tolist()  # 0-based
print(f"Rows with missing host_years: {len(missing_idx)}")
print("0-based row indices:", missing_idx)
print("1-based row indices:", [i + 1 for i in missing_idx])

if missing_idx:
    print("\nSample of missing rows:")
    print(df.loc[missing_idx, ['id', 'host_years']])


# In[ ]:


import pandas as pd

src = 'listing_honolulu.csv'
out = 'listing_honolulu2.csv'

df = pd.read_csv(src)
after_drop = df[~df['host_years'].isna()]
removed = len(df) - len(after_drop)

after_drop.to_csv(out, index=False)
print(f"Dropped rows with missing host_years: {removed}; remaining {len(after_drop)} rows")


# In[ ]:


import pandas as pd
from pathlib import Path

hon_path = 'listing_honolulu2.csv'
report_path = Path('listing_honolulu_report2.txt')

df = pd.read_csv(hon_path)

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

with report_path.open('w', encoding='utf-8') as f:
    f.write(f"File: {hon_path}\n")
    f.write(f"Rows: {len(df)}, Columns: {df.shape[1]}\n\n")
    f.write("Column dtypes:\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nMissing value summary:\n")
    for col in df.columns:
        f.write(f"{col}: {missing[col]} missing ({missing_pct[col]}%)\n")

print("Report written to:", report_path.resolve())
print("(Open the file to view full output)")


# In[ ]:


import pandas as pd

path = 'listing_honolulu2.csv'  # change path if you want to inspect another file
df = pd.read_csv(path)

missing = df['host_is_superhost'].isna().sum()
value_counts = df['host_is_superhost'].value_counts(dropna=False)

print(f"File: {path}")
print(f"Missing values: {missing}")
print("\nValue counts (including missing):")
print(value_counts)


# In[ ]:


import pandas as pd

src = 'listing_honolulu2.csv'
out = 'listing_honolulu3.csv'

df = pd.read_csv(src)

def to_bool(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    if s == 't':
        return True
    if s == 'f':
        return False
    return False

df['host_is_superhost'] = df['host_is_superhost'].apply(to_bool).astype(bool)

df.to_csv(out, index=False)
print(f"Converted and saved to {out}")
print(df['host_is_superhost'].value_counts(dropna=False))


# In[ ]:


import pandas as pd
from pathlib import Path

hon_path = 'listing_honolulu3.csv'
report_path = Path('listing_honolulu_report3.txt')

df = pd.read_csv(hon_path)

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

with report_path.open('w', encoding='utf-8') as f:
    f.write(f"File: {hon_path}\n")
    f.write(f"Rows: {len(df)}, Columns: {df.shape[1]}\n\n")
    f.write("Column dtypes:\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nMissing value summary:\n")
    for col in df.columns:
        f.write(f"{col}: {missing[col]} missing ({missing_pct[col]}%)\n")

print("Report written to:", report_path.resolve())
print("(Open the file to view full output)")


# In[ ]:


import pandas as pd

src = 'listing_honolulu3.csv'
out = 'listing_honolulu4.csv'

df = pd.read_csv(src)
pct_cols = ['host_response_rate', 'host_acceptance_rate']

for col in pct_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace('%', '').str.strip(),
            errors='coerce'
        ) / 100

df.to_csv(out, index=False)
print(f"Converted {pct_cols} and saved to {out}")
print(df[pct_cols].head())


# In[ ]:


import pandas as pd
from pathlib import Path

hon_path = 'listing_honululu4.csv'
report_path = Path('listing_honululu_report4.txt')

df = pd.read_csv(hon_path)

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

with report_path.open('w', encoding='utf-8') as f:
    f.write(f"File: {hon_path}\n")
    f.write(f"Rows: {len(df)}, Columns: {df.shape[1]}\n\n")
    f.write("Column dtypes:\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nMissing value summary:\n")
    for col in df.columns:
        f.write(f"{col}: {missing[col]} missing ({missing_pct[col]}%)\n")

print("Report written to:", report_path.resolve())
print("(Open the file to view full output)")


# In[ ]:


import pandas as pd
from pathlib import Path

src = "listing_honolulu4.csv"
out = "listing_honolulu5.csv"
report_path = Path("listing_honolulu5_report.txt")

df = pd.read_csv(src)

score_cols = [
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_communication",
]
rate_cols = ["host_response_rate", "host_acceptance_rate"]
count_cols_zero = ["reviews_per_month"]

bool_cols = [c for c in df.columns if df[c].dtype == bool]
num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in bool_cols]

report = []

def add_indicator(col):
    ind_col = f"{col}_missing"
    df[ind_col] = df[col].isna().astype(int)
    report.append((col, df[col].dtype, df[col].isna().sum(), "added missing indicator"))

for col in score_cols + rate_cols + count_cols_zero:
    if col in df.columns:
        add_indicator(col)

for col in score_cols:
    if col in df.columns:
        miss = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        report.append((col, df[col].dtype, miss, "filled NaN with 0 (no score)"))

for col in rate_cols:
    if col in df.columns:
        miss = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        report.append((col, df[col].dtype, miss, "filled NaN with 0 (inactive)"))

for col in count_cols_zero:
    if col in df.columns:
        miss = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        report.append((col, df[col].dtype, miss, "filled NaN with 0 (no reviews flow)"))

remain_num = [c for c in num_cols if c not in score_cols + rate_cols + count_cols_zero]
for col in remain_num:
    miss = df[col].isna().sum()
    df[col] = df[col].fillna(df[col].median())
    report.append((col, df[col].dtype, miss, "filled with median"))

cat_cols = [c for c in df.select_dtypes(include=["object"]).columns]
for col in cat_cols:
    miss = df[col].isna().sum()
    df[col] = df[col].fillna("missing")
    report.append((col, df[col].dtype, miss, "filled with 'missing'") )

df.to_csv(out, index=False)

with report_path.open("w", encoding="utf-8") as f:
    f.write("Processing report (column, dtype, missing count, handling)\n")
    for col, dt, miss, how in report:
        f.write(f"{col}: {dt}, missing={miss}, {how}\n")

print(f"Done: saved {out}")
print(f"Full report written to: {report_path.resolve()}")
print("Example indicator columns:", [c for c in df.columns if c.endswith('_missing')][:10])


# In[ ]:


import pandas as pd
from pathlib import Path

hon_path = 'listing_honolulu5.csv'
report_path = Path('listing_honolulu_report6.txt')

df = pd.read_csv(hon_path)

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

with report_path.open('w', encoding='utf-8') as f:
    f.write(f"File: {hon_path}\n")
    f.write(f"Rows: {len(df)}, Columns: {df.shape[1]}\n\n")
    f.write("Column dtypes:\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nMissing value summary:\n")
    for col in df.columns:
        f.write(f"{col}: {missing[col]} missing ({missing_pct[col]}%)\n")

print("Report written to:", report_path.resolve())
print("(Open the file to view full output)")


# In[ ]:


# Sample a small subset (first 100 rows) to compute driving distance/time to HNL and Waikiki
# Ensure googlemaps is installed before running: !pip install googlemaps
import pandas as pd
import googlemaps
from datetime import datetime

api_key = "AIzaSyBIn3VvUd2Uc6nrUHS7cn446nxFKdE90Qg"  # TODO: provide a valid key
sample_size = 100  # adjustable; start with 50-100 to validate

# Destination coordinates
DEST_HNL = (21.3245, -157.9251)   # Honolulu Intl Airport (HNL)
DEST_WK = (21.2810, -157.8370)    # Waikiki Beach

src_path = "/Users/jiangzhanyuan/Desktop/second year/IEOR242A/Final Project/Final Data/listing_honolulu5.csv"
out_path = "listing_honolulu5_sample_routes.csv"

full_df = pd.read_csv(src_path)
df = full_df.head(sample_size).copy()

for col in ["drive_dist_hnl_km", "drive_time_hnl_min", "drive_dist_wk_km", "drive_time_wk_min"]:
    df[col] = pd.NA

client = googlemaps.Client(key=api_key)

def query_route(origin, dest):
    try:
        res = client.directions(origin=origin, destination=dest, mode="driving", departure_time=datetime.now())
        if not res:
            return None, None
        leg = res[0]["legs"][0]
        dist_km = leg["distance"]["value"] / 1000.0
        dur_min = leg["duration"]["value"] / 60.0
        return dist_km, dur_min
    except Exception as e:
        print("route error", e)
        return None, None

for i, row in df.iterrows():
    lat, lon = row.get("latitude"), row.get("longitude")
    if pd.isna(lat) or pd.isna(lon):
        continue
    origin = (lat, lon)

    d_km, t_min = query_route(origin, DEST_HNL)
    df.at[i, "drive_dist_hnl_km"] = d_km
    df.at[i, "drive_time_hnl_min"] = t_min

    d_km, t_min = query_route(origin, DEST_WK)
    df.at[i, "drive_dist_wk_km"] = d_km
    df.at[i, "drive_time_wk_min"] = t_min

print("Sample calculations complete, saving...")
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}, sample rows: {len(df)}")



# In[ ]:


get_ipython().system('pip install googlemaps')


# In[ ]:


import pandas as pd
import googlemaps
import time
import math
from datetime import datetime
from pathlib import Path

# === Configuration ===
api_key = "AIzaSyBIn3VvUd2Uc6nrUHS7cn446nxFKdE90Qg"  # TODO: provide your valid key
src_path = "/Users/jiangzhanyuan/Desktop/second year/IEOR242A/Final Project/Final Data/listing_honolulu5.csv"
from pathlib import Path
checkpoint_path = Path("/Users/jiangzhanyuan/Desktop/second year/IEOR242A/Final Project/Final Data/Checkpoints/listing_honolulu5_routes_checkpoint.csv")
out_path = "FFFFFFFFFlisting_honolulu5_routes_full.csv"
batch_size = 200       # rows per batch
sleep_sec = 3          # pause between batches to avoid rate limits
coord_round = 4        # round coords to reduce duplicate requests

# Destination coordinates
DEST_HNL = (21.3245, -157.9251)   # Honolulu Intl Airport
DEST_WK  = (21.2810, -157.8370)   # Waikiki Beach

# === Prepare data ===
df = pd.read_csv(src_path)
route_cols = ["drive_dist_hnl_km", "drive_time_hnl_min", "drive_dist_wk_km", "drive_time_wk_min"]
for col in route_cols:
    if col not in df.columns:
        df[col] = pd.NA

# Load checkpoint if it exists (resume capability)
if checkpoint_path.exists():
    ckpt = pd.read_csv(checkpoint_path)
    df.update(ckpt)
    print(f"Loaded checkpoint: {checkpoint_path}")

client = googlemaps.Client(key=api_key)

# Coordinate cache to avoid duplicate requests
cache = {}
def norm_coord(lat, lon, ndigits=coord_round):
    return (round(lat, ndigits), round(lon, ndigits))

def query_route(origin, dest):
    try:
        res = client.directions(origin=origin, destination=dest, mode="driving", departure_time=datetime.now())
        if not res:
            return math.nan, math.nan
        leg = res[0]["legs"][0]
        dist_km = leg["distance"]["value"] / 1000.0
        dur_min = leg["duration"]["value"] / 60.0
        return dist_km, dur_min
    except Exception as e:
        print("route error:", e)
        return math.nan, math.nan

# Find rows still missing route info
unfinished_idx = df[df[route_cols].isna().any(axis=1)].index.tolist()
print(f"Remaining rows: {len(unfinished_idx)} / {len(df)}")

for start in range(0, len(unfinished_idx), batch_size):
    batch_idx = unfinished_idx[start:start+batch_size]
    print(f"Processing rows {start} - {start + len(batch_idx) - 1} (batch size {len(batch_idx)})")

    for i in batch_idx:
        lat, lon = df.at[i, "latitude"], df.at[i, "longitude"]
        if pd.isna(lat) or pd.isna(lon):
            continue
        key = norm_coord(lat, lon)
        if key in cache:
            d_h_km, d_h_min, d_w_km, d_w_min = cache[key]
        else:
            d_h_km, d_h_min = query_route(key, DEST_HNL)
            d_w_km, d_w_min = query_route(key, DEST_WK)
            cache[key] = (d_h_km, d_h_min, d_w_km, d_w_min)

        df.at[i, "drive_dist_hnl_km"] = d_h_km
        df.at[i, "drive_time_hnl_min"] = d_h_min
        df.at[i, "drive_dist_wk_km"] = d_w_km
        df.at[i, "drive_time_wk_min"] = d_w_min

    # Write checkpoint each batch for resumability
    df.to_csv(checkpoint_path, index=False)
    print(f"Batch complete, checkpoint written: {checkpoint_path}")
    time.sleep(sleep_sec)

# Save final file
df.to_csv(out_path, index=False)
print(f"All done, saved {out_path}")


# In[ ]:


import pandas as pd

src = "FFFFFFFFFlisting_honolulu5_routes_full.csv"
out = "Lisiting_Honolulu.csv"
cols_to_round = [
    "drive_dist_hnl_km",
    "drive_time_hnl_min",
    "drive_dist_wk_km",
    "drive_time_wk_min",
]

df = pd.read_csv(src)
for c in cols_to_round:
    if c in df.columns:
        df[c] = df[c].round(1)

df.to_csv(out, index=False)
print(f"Rounded four route columns to 1 decimal and saved as {out}")
print(df[cols_to_round].head())


# In[ ]:


import pandas as pd
from pathlib import Path

hon_path = 'Lisiting_Honolulu.csv'
report_path = Path('listing_honolulu_report7.txt')

df = pd.read_csv(hon_path)

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

with report_path.open('w', encoding='utf-8') as f:
    f.write(f"File: {hon_path}\n")
    f.write(f"Rows: {len(df)}, Columns: {df.shape[1]}\n\n")
    f.write("Column dtypes:\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nMissing value summary:\n")
    for col in df.columns:
        f.write(f"{col}: {missing[col]} missing ({missing_pct[col]}%)\n")

print("Report written to:", report_path.resolve())
print("(Open the file to view full output)")


# In[ ]:


# View value counts for instant_bookable and host_has_profile_pic
import pandas as pd

path = "Lisiting_Honolulu.csv"
df = pd.read_csv(path)

for col in ["instant_bookable", "host_has_profile_pic"]:
    if col not in df.columns:
        print(f"Column {col} not found in {path}")
        continue
    vc = df[col].value_counts(dropna=False)
    print(f"\n{col} value counts (including missing):")
    print(vc)


# In[ ]:


# Convert instant_bookable and host_has_profile_pic to booleans (t/f -> True/False, others/NaN -> False)
import pandas as pd

path = "Lisiting_Honolulu.csv"
df = pd.read_csv(path)

def to_bool_tf(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    if s == 't':
        return True
    if s == 'f':
        return False
    return False

for col in ["instant_bookable", "host_has_profile_pic"]:
    if col in df.columns:
        df[col] = df[col].apply(to_bool_tf).astype(bool)

df.to_csv(path, index=False)

for col in ["instant_bookable", "host_has_profile_pic"]:
    if col in df.columns:
        print(f"{col} -> True/False counts:")
        print(df[col].value_counts(dropna=False))



# In[ ]:


# Inspect value counts for room_type / neighbourhood_group / neighbourhood
import pandas as pd

path = "Lisiting_Honolulu.csv"
df = pd.read_csv(path)

for col in ["room_type", "neighbourhood_group", "neighbourhood"]:
    if col not in df.columns:
        print(f"Column {col} not found in {path}")
        continue
    vc = df[col].value_counts(dropna=False)
    print(f"\n{col} value counts (including missing):")
    print(vc)


# In[ ]:


import pandas as pd

path = "Lisiting_Honolulu.csv"
df = pd.read_csv(path)

for col in ["host_response_time"]:
    if col not in df.columns:
        print(f"Column {col} not found in {path}")
        continue
    vc = df[col].value_counts(dropna=False)
    print(f"\n{col} value counts (including missing):")
    print(vc)


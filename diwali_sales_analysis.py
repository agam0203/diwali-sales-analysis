import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

np.random.seed(42)
os.makedirs("/home/claude/diwali_project/charts", exist_ok=True)

# ── 1. GENERATE DATASET ──────────────────────────────────────────────
n = 1000

states = ['Uttar Pradesh','Maharashtra','Delhi','Karnataka','Gujarat',
          'Rajasthan','West Bengal','Tamil Nadu','Madhya Pradesh','Punjab']
state_weights = [0.18,0.16,0.14,0.12,0.10,0.08,0.07,0.07,0.05,0.03]

categories = ['Electronics','Clothing','Home Decor','Food & Sweets',
              'Toys & Games','Jewellery','Beauty & Personal Care','Sports']
cat_weights = [0.22,0.20,0.15,0.13,0.10,0.08,0.07,0.05]

occupations = ['IT Professional','Govt Employee','Business Owner',
               'Student','Homemaker','Healthcare','Teacher']
occ_weights  = [0.22,0.18,0.16,0.14,0.12,0.10,0.08]

base_prices = {
    'Electronics':15000,'Clothing':2500,'Home Decor':3500,
    'Food & Sweets':800,'Toys & Games':1200,'Jewellery':8000,
    'Beauty & Personal Care':1500,'Sports':3000
}

genders     = np.random.choice(['Male','Female'], n, p=[0.55,0.45])
age_groups  = np.random.choice(['18-25','26-35','36-45','46-55','55+'], n,
                                p=[0.20,0.35,0.25,0.12,0.08])
states_col  = np.random.choice(states, n, p=state_weights)
categories_col = np.random.choice(categories, n, p=cat_weights)
occupations_col = np.random.choice(occupations, n, p=occ_weights)
marital     = np.random.choice(['Married','Unmarried'], n, p=[0.60,0.40])

amounts = []
for cat in categories_col:
    base  = base_prices[cat]
    noise = np.random.uniform(0.7, 1.5)
    amounts.append(round(base * noise, -1))   # round to nearest 10

orders = np.random.randint(1, 6, n)
ratings= np.random.choice([3,4,5], n, p=[0.15,0.45,0.40])

df = pd.DataFrame({
    'Customer_ID'  : [f'CID{str(i).zfill(4)}' for i in range(1, n+1)],
    'Gender'       : genders,
    'Age_Group'    : age_groups,
    'State'        : states_col,
    'Marital_Status': marital,
    'Occupation'   : occupations_col,
    'Product_Category': categories_col,
    'Orders'       : orders,
    'Amount'       : amounts,
    'Rating'       : ratings
})

df['Total_Revenue'] = df['Amount'] * df['Orders']
df.to_csv('/home/claude/diwali_project/diwali_sales.csv', index=False)
print("Dataset created:", df.shape)
print(df.head(3))

# ── PLOT STYLE ────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
TITLE_FS  = 14
LABEL_FS  = 11
TICK_FS   = 9

def save(name):
    plt.tight_layout()
    plt.savefig(f'/home/claude/diwali_project/charts/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {name}.png")

# ── CHART 1: Revenue by Category ─────────────────────────────────────
cat_rev = df.groupby('Product_Category')['Total_Revenue'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(cat_rev.index, cat_rev.values,
              color=sns.color_palette('Blues_d', len(cat_rev)))
ax.set_title('Total Revenue by Product Category', fontsize=TITLE_FS, fontweight='bold')
ax.set_xlabel('Category', fontsize=LABEL_FS)
ax.set_ylabel('Total Revenue (₹)', fontsize=LABEL_FS)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
ax.tick_params(axis='x', rotation=30, labelsize=TICK_FS)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20000,
            f'₹{bar.get_height()/1e6:.1f}M', ha='center', va='bottom', fontsize=8)
save('01_revenue_by_category')

# ── CHART 2: Revenue by State (Top 7) ────────────────────────────────
state_rev = df.groupby('State')['Total_Revenue'].sum().sort_values(ascending=False).head(7)
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(x=state_rev.index, y=state_rev.values,
            palette='Oranges_d', ax=ax)
ax.set_title('Top 7 States by Revenue', fontsize=TITLE_FS, fontweight='bold')
ax.set_xlabel('State', fontsize=LABEL_FS)
ax.set_ylabel('Total Revenue (₹)', fontsize=LABEL_FS)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
ax.tick_params(axis='x', rotation=25, labelsize=TICK_FS)
save('02_revenue_by_state')

# ── CHART 3: Gender-wise Buying Behaviour ────────────────────────────
gender_data = df.groupby(['Gender','Product_Category'])['Total_Revenue'].sum().unstack()
gender_data.plot(kind='bar', figsize=(11,5), colormap='Set2')
plt.title('Gender-wise Spending Across Categories', fontsize=TITLE_FS, fontweight='bold')
plt.xlabel('Gender', fontsize=LABEL_FS)
plt.ylabel('Total Revenue (₹)', fontsize=LABEL_FS)
plt.legend(title='Category', bbox_to_anchor=(1.01,1), fontsize=8)
plt.xticks(rotation=0)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
save('03_gender_category_spending')

# ── CHART 4: Age Group vs Revenue ────────────────────────────────────
age_order = ['18-25','26-35','36-45','46-55','55+']
age_rev   = df.groupby('Age_Group')['Total_Revenue'].sum().reindex(age_order)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=age_rev.index, y=age_rev.values, palette='Purples_d', ax=ax)
ax.set_title('Revenue by Age Group', fontsize=TITLE_FS, fontweight='bold')
ax.set_xlabel('Age Group', fontsize=LABEL_FS)
ax.set_ylabel('Total Revenue (₹)', fontsize=LABEL_FS)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
save('04_revenue_by_age_group')

# ── CHART 5: Occupation-wise Spending ────────────────────────────────
occ_rev = df.groupby('Occupation')['Total_Revenue'].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(occ_rev.index, occ_rev.values,
        color=sns.color_palette('Greens_d', len(occ_rev)))
ax.set_title('Revenue by Occupation', fontsize=TITLE_FS, fontweight='bold')
ax.set_xlabel('Total Revenue (₹)', fontsize=LABEL_FS)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
save('05_revenue_by_occupation')

# ── CHART 6: Marital Status vs Avg Order Value ───────────────────────
marital_avg = df.groupby(['Marital_Status','Gender'])['Amount'].mean().unstack()
marital_avg.plot(kind='bar', figsize=(7,5), color=['#E07B8A','#5B9BD5'])
plt.title('Avg Order Value: Marital Status & Gender', fontsize=TITLE_FS, fontweight='bold')
plt.xlabel('Marital Status', fontsize=LABEL_FS)
plt.ylabel('Avg Amount (₹)', fontsize=LABEL_FS)
plt.xticks(rotation=0)
plt.legend(title='Gender')
save('06_marital_gender_avg_order')

# ── CHART 7: Rating Distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,4))
df['Rating'].value_counts().sort_index().plot(kind='bar', color=['#f4a261','#2a9d8f','#264653'], ax=ax)
ax.set_title('Customer Rating Distribution', fontsize=TITLE_FS, fontweight='bold')
ax.set_xlabel('Rating (out of 5)', fontsize=LABEL_FS)
ax.set_ylabel('Number of Customers', fontsize=LABEL_FS)
ax.tick_params(axis='x', rotation=0)
save('07_rating_distribution')

# ── SUMMARY STATS ─────────────────────────────────────────────────────
print("\n===== KEY INSIGHTS =====")
print(f"Total Revenue       : ₹{df['Total_Revenue'].sum():,.0f}")
print(f"Total Orders        : {df['Orders'].sum():,}")
print(f"Avg Order Value     : ₹{df['Amount'].mean():,.0f}")
print(f"Top Category        : {cat_rev.idxmax()} (₹{cat_rev.max():,.0f})")
print(f"Top State           : {state_rev.idxmax()}")
print(f"Highest Spending Age: {df.groupby('Age_Group')['Total_Revenue'].sum().idxmax()}")
print(f"Top Occupation      : {df.groupby('Occupation')['Total_Revenue'].sum().idxmax()}")

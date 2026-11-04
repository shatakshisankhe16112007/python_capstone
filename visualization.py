import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")

os.makedirs("output_graphs", exist_ok=True)

df = pd.read_csv("dataset/crop_yield.csv")

df.columns = df.columns.str.lower().str.strip()

numeric_columns = [ "year", "area", "production", "fertilizer", "pesticide", "yield"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["crop", "year", "season", "state","area", "production", "yield"])

for col in ["area", "production", "fertilizer","pesticide", "yield"]:
    df = df[(df[col].isna()) | (df[col] >= 0)]

df = df.drop_duplicates()

region_mapping = {
    "North": ["Punjab", "Haryana", "Himachal Pradesh",
        "Jammu and Kashmir", "Uttar Pradesh",
        "Uttarakhand", "Delhi", "Ladakh", "Rajasthan"],

    "South": ["Andhra Pradesh", "Karnataka", "Kerala",
        "Tamil Nadu", "Telangana", "Puducherry"],

    "East": ["Bihar", "Jharkhand", "Odisha",
        "West Bengal"],

    "West": ["Goa", "Gujarat", "Maharashtra"],

    "Central": ["Madhya Pradesh", "Chhattisgarh"],

    "Northeast": ["Assam", "Arunachal Pradesh", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland","Sikkim", "Tripura"]
}

state_to_region = {}

for region, states in region_mapping.items():
    for state in states:
        state_to_region[state.lower()] = region

df["region"] = df["state"].str.strip().str.lower().map(state_to_region)

unmapped_states = df[df["region"].isna()]["state"].unique()

if len(unmapped_states) > 0:
    print("Unmapped states:", unmapped_states)

df = df.dropna(subset=["region"])

print("Dataset cleaned successfully!")
print("Final dataset shape:", df.shape)

# FIGURE 1: REGIONAL CROP PRODUCTION ANALYSIS
fig, axes = plt.subplots(2, 2, figsize=(18, 13))

fig.patch.set_facecolor("lightblue")

for ax in axes.flat:
    ax.set_facecolor("white")

fig.suptitle("REGIONAL CROP PRODUCTION ANALYSIS",
    fontsize=20,fontweight="bold")

# Graph 1: Total production by region

region_production = df.groupby("region")["production"].sum()
region_production = region_production.sort_values()

axes[0, 0].barh(region_production.index,
    region_production.values,
    color=sns.color_palette("viridis", len(region_production)))

axes[0, 0].set_title("Total Crop Production by Region")
axes[0, 0].set_xlabel("Total Production")
axes[0, 0].set_ylabel("Region")

# Graph 2: Top 6 crops production across regions

top_crops = (df.groupby("crop")["production"].sum()
    .nlargest(6).index)

crop_data = df[df["crop"].isin(top_crops)]

crop_region = crop_data.pivot_table(index="region",columns="crop",
    values="production",aggfunc="sum",fill_value=0)

crop_region.plot(kind="bar",stacked=True,ax=axes[0, 1],colormap="Set3")

axes[0, 1].set_title("Top 6 Crops Production Across Regions")
axes[0, 1].set_xlabel("Region")
axes[0, 1].set_ylabel("Production")
axes[0, 1].legend(title="Crops", fontsize=8,loc="upper left",bbox_to_anchor=(1, 1))

# Graph 3: Heatmap of average crop production

average_production = crop_data.pivot_table(index="region",columns="crop",
    values="production",aggfunc="mean")

sns.heatmap(average_production,annot=True,fmt=".1f",
    cmap="YlGnBu",linewidths=0.5,ax=axes[1, 0])

axes[1, 0].set_title("Average Crop Production Heatmap")
axes[1, 0].set_xlabel("Crop")
axes[1, 0].set_ylabel("Region")

# Graph 4: Production by region and season with error bars
season_data = df.groupby(["region", "season"]
)["production"].agg(["mean", "std"]).reset_index()

regions = season_data["region"].unique()
seasons = season_data["season"].unique()

x = np.arange(len(regions))
width = 0.8 / len(seasons)

for i, season in enumerate(seasons):
    data = season_data[season_data["season"] == season]
    values = data.set_index("region").reindex(regions)
    axes[1, 1].bar(x + i * width,values["mean"],width,
        yerr=values["std"].fillna(0),capsize=4,label=season)

axes[1, 1].set_title("Average Production by Region and Season")
axes[1, 1].set_xlabel("Region")
axes[1, 1].set_ylabel("Average Production")

axes[1, 1].set_xticks(x + width * (len(seasons) - 1) / 2)
axes[1, 1].set_xticklabels(regions)

axes[1, 1].legend(title="Season")

# FIGURE 2: AGRICULTURAL FACTOR RELATIONSHIPS

fig, axes = plt.subplots(2, 2, figsize=(18, 13))

fig.patch.set_facecolor("lightblue")

for ax in axes.flat:
    ax.set_facecolor("white")

fig.suptitle("AGRICULTURAL FACTOR RELATIONSHIPS",fontsize=20,
    fontweight="bold")

sample_data = df.sample( n=min(2000, len(df)), random_state=42)

# Graph 1: Fertilizer vs yield

fertilizer_data = sample_data.dropna(subset=["fertilizer", "yield"])

sns.scatterplot(data=fertilizer_data,x="fertilizer",y="yield",
    hue="region", palette="viridis", ax=axes[0, 0], alpha=0.7)

sns.regplot(data=fertilizer_data,x="fertilizer",y="yield",
    scatter=False,color="red",ax=axes[0, 0])

axes[0, 0].set_title("Relationship Between Fertilizer and Yield")
axes[0, 0].set_xlabel("Fertilizer")
axes[0, 0].set_ylabel("Crop Yield")
axes[0, 0].legend(title="Region", fontsize=8)

# Graph 2: Pesticide vs yield

pesticide_data = sample_data.dropna(subset=["pesticide", "yield"])

sns.scatterplot(data=pesticide_data,x="pesticide",y="yield",hue="region",
    palette="Set1",ax=axes[0, 1],alpha=0.7)

sns.regplot(data=pesticide_data,x="pesticide",y="yield",
    scatter=False,color="black",ax=axes[0, 1])

axes[0, 1].set_title("Relationship Between Pesticide and Yield")
axes[0, 1].set_xlabel("Pesticide")
axes[0, 1].set_ylabel("Crop Yield")
axes[0, 1].legend(title="Region", fontsize=8)

# Graph 3: Area vs production bubble chart

sns.scatterplot(data=sample_data,x="area",y="production",hue="region",size="yield",
    sizes=(20, 200),palette="Dark2",ax=axes[1, 0],alpha=0.7)

axes[1, 0].set_title("Cultivated Area vs Crop Production")
axes[1, 0].set_xlabel("Cultivated Area")
axes[1, 0].set_ylabel("Crop Production")
axes[1, 0].legend(title="Region and Yield",fontsize=7,
    loc="upper left",bbox_to_anchor=(1, 1))

# Graph 4: Correlation heatmap

correlation = df[
    ["area", "production", "fertilizer", "pesticide", "yield"]
].corr()

sns.heatmap(correlation,annot=True,cmap="coolwarm",fmt=".2f",
    linewidths=0.5,ax=axes[1, 1])

axes[1, 1].set_title("Correlation Between Agricultural Factors")

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("output_graphs/figure2_agricultural_relationships.png",dpi=300)
plt.show()
plt.close()

# FIGURE 3: CROP YIELD AND SEASONAL ANALYSIS
fig, axes = plt.subplots(2, 2, figsize=(18, 13))

fig.patch.set_facecolor("lightblue")

for ax in axes.flat:
    ax.set_facecolor("white")

fig.suptitle("CROP YIELD AND SEASONAL ANALYSIS",fontsize=20,fontweight="bold")

# Graph 1: Yield distribution across regions

sns.boxplot(data=df,x="region",y="yield",hue="region",palette="Set2",
    ax=axes[0, 0], showfliers=False, legend=False)

axes[0, 0].set_title("Crop Yield Distribution Across Regions")
axes[0, 0].set_xlabel("Region")
axes[0, 0].set_ylabel("Crop Yield")

# Graph 2: Yield distribution across seasons

sns.violinplot(data=df,x="season",y="yield",hue="season",palette="Pastel1",
    ax=axes[0, 1],cut=0,inner="quart",legend=False)

axes[0, 1].set_title("Crop Yield Distribution Across Seasons")
axes[0, 1].set_xlabel("Season")
axes[0, 1].set_ylabel("Crop Yield")

# Graph 3: Yearly average yield trends across regions
yearly_yield = df.groupby(["year", "region"]
)["yield"].agg(["mean", "std"]).reset_index()

for region in yearly_yield["region"].unique():
    data = yearly_yield[yearly_yield["region"] == region]
    axes[1, 0].errorbar(data["year"], data["mean"], yerr=data["std"].fillna(0),
        marker="o", capsize=4, label=region)

axes[1, 0].set_title("Yearly Average Crop Yield Trends")
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("Average Crop Yield")

axes[1, 0].legend(title="Region")

# Graph 4: Yield across regions and seasons

sns.boxplot( data=df, x="region", y="yield", hue="season", palette="Spectral",
    ax=axes[1, 1],showfliers=False)

axes[1, 1].set_title("Crop Yield Across Regions and Seasons")
axes[1, 1].set_xlabel("Region")
axes[1, 1].set_ylabel("Crop Yield")
axes[1, 1].legend(title="Season",fontsize=8,loc="upper left",bbox_to_anchor=(1, 1))

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("output_graphs/figure3_yield_seasonal_analysis.png",dpi=300)
plt.show()
plt.close()
print("\nAll three dashboards have been generated successfully!")
print("Check the output_graphs folder.")
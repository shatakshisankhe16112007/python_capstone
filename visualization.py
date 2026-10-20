import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("output_graphs", exist_ok=True)

df = pd.read_csv("dataset/crop_yield.csv")

df = df.dropna(subset=["crop", "season", "yield"]).copy()

df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
df = df.dropna(subset=["yield"])
df = df[df["yield"] >= 0]

sns.set_theme(style="whitegrid")


# GRAPH 1: Average Crop Yield by Crop
average_yield = (df.groupby("crop")["yield"].mean() .sort_values())

plt.figure(figsize=(12, 10))

average_yield.plot(kind="barh")

plt.title("Average Crop Yield by Crop", fontsize=15)
plt.xlabel("Average Crop Yield")
plt.ylabel("Crop")
plt.tight_layout()

plt.savefig( "output_graphs/average_yield_by_crop.png",dpi=300,bbox_inches="tight")

plt.show()
plt.close()


# GRAPH 2: Crop Yield by Season
plt.figure(figsize=(10, 6))

sns.boxplot(data=df[df["yield"] > 0],x="season",y="yield")

plt.yscale("log")

plt.title("Crop Yield Distribution by Season", fontsize=15)
plt.xlabel("Season")
plt.ylabel("Crop Yield (log scale)")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("output_graphs/crop_yield_by_season.png",dpi=300,bbox_inches="tight")

plt.show()
plt.close()


# GRAPH 3: Crop Yield Distribution
positive_yield = df[df["yield"] > 0]["yield"]

plt.figure(figsize=(10, 6))

plt.hist(positive_yield,bins=30,log=True,edgecolor="black")

plt.title("Distribution of Positive Crop Yield Values", fontsize=15)
plt.xlabel("Crop Yield")
plt.ylabel("Frequency (log scale)")

plt.tight_layout()

plt.savefig("output_graphs/crop_yield_distribution.png",dpi=300,bbox_inches="tight")

plt.show()
plt.close()


print("All three graphs generated successfully!")
print("Check the output_graphs folder.")
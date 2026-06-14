import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("customers.xlsx")
print("Customer Data:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

plt.hist(df["Age"], bins=5)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

plt.hist(df["Annual Income($)"], bins=5)
plt.title("Annual Income Distribution")
plt.xlabel("Income")
plt.ylabel("Number of Customers")
plt.show()

plt.hist(df["Spending Score"], bins=5)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.show()

from sklearn.cluster import KMeans

X=df[["Annual Income($)", "Spending Score"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

print("\nCustomer Clusters:")
print(df[["CustomerID", "Annual Income($)", "Spending Score", "Cluster"]])

plt.figure(figsize=(8,6))

plt.scatter(
  df["Annual Income($)"],
  df["Spending Score"],
  c=df["Cluster"]
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.show()
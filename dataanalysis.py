import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

file = pd.read_csv('Time_Wasters_on_Social_Media.csv')

# Convert boolean-like columns
file["Owns Property"] = file["Owns Property"].astype(bool)

# Convert categorical columns to category type
categorical_columns = ["Gender", "Location", "Profession", "Platform", "DeviceType", "OS", "Watch Reason", "CurrentActivity", "ConnectionType", "Video Category"]
file[categorical_columns] = file[categorical_columns].astype("category")

# Convert Watch Time into hours
file["Watch Hour"] = pd.to_datetime(file["Watch Time"], format="%I:%M %p", errors="coerce").dt.hour

# Convert numerical columns to appropriate types
numeric_columns = ["Age", "Income", "Debt", "Total Time Spent", "Number of Sessions", "Video Length",
                   "Engagement", "Importance Score", "Time Spent On Video", "Number of Videos Watched",
                   "Scroll Rate", "ProductivityLoss", "Satisfaction", "Self Control", "Addiction Level"]
file[numeric_columns] = file[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Check the first few rows
print(file.head())

# Correlation Heat Map
correlation_matrix = file[numeric_columns].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sea.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Stacked Bar Chart
watch_reason_addiction = file.groupby("Watch Reason")["Addiction Level"].mean().sort_values()

# Plot
watch_reason_addiction.plot(kind="bar", color="skyblue")
plt.title("Watch Reason vs. Average Addiction Level")
plt.xlabel("Watch Reason")
plt.ylabel("Average Addiction Level")
plt.xticks(rotation=45)
plt.show()

# line graph
sea.lineplot(data=file, x="Watch Hour", y="ProductivityLoss")
plt.title("Watch Hour vs. Productivity Loss")
plt.xlabel("Hour of the Day")
plt.ylabel("Productivity Loss")
plt.show()

#Scatterplots - 
plt.figure(figsize=(8,5))
sea.scatterplot(x=file['Total Time Spent'], y=file['Addiction Level'])
plt.title("Total Time Spent vs Addiction Level")
plt.xlabel("Total Time Spent")
plt.ylabel("Addiction Level")
plt.show()

#Scatterplots - 
fig, axs = plt.subplots(2, 2, figsize=(13, 13))

sea.stripplot(data=file, x='Total Time Spent', y='Addiction Level', ax=axs[0, 0])
axs[0, 0].set_title("Total Time Spent vs Addiction Level")
axs[0, 0].set_xlabel("Total Time Spent")
axs[0, 0].set_xticks(np.linspace(10, 268, 10))
axs[0, 0].set_ylabel("Addiction Level")

sea.stripplot(data=file, x='Scroll Rate', y='Addiction Level', ax=axs[0, 1])
axs[0, 1].set_title("Scroll Rate vs. Addiction Level")
axs[0, 1].set_xlabel("Scroll Rate")
axs[0, 1].set_xticks(np.linspace(1, 100, 10))
axs[0, 1].set_ylabel("Addiction Level")


sea.stripplot(x=file['Number of Sessions'], y=file['Addiction Level'], ax=axs[1, 0])
axs[1, 0].set_title("Number of Sessions vs. Addiction Level")
axs[1, 0].set_xlabel("Number of Sessions")
axs[1, 0].set_ylabel("Addiction Level")


sea.scatterplot(data=file, x='Number of Videos Watched', y='Addiction Level', ax=axs[1,1])
axs[1, 1].set_title("Number of Videos Watched vs. Addiction Level")
axs[1, 1].set_xlabel("Number of Videos Watched")
axs[1, 1].set_ylabel("Addiction Level")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sea.stripplot(data=file, x="Self Control", y="Addiction Level", ax=axs[0])
axs[0].set_title("Self Control vs. Addiction")

sea.stripplot(data=file, x='Satisfaction', y='Addiction Level', jitter=True, alpha=0.7, size=3, ax=axs[1])
axs[1].set_title("Satisfaction vs. Addiction")

sea.stripplot(data=file, x="ProductivityLoss", y="Addiction Level", ax=axs[2])
axs[2].set_title("Productivity Loss vs. Addiction Level")




plt.show()

#Box Plots

fig, axs = plt.subplots(2, 3, figsize=(15,11))
def determine_level(income):
  if income <= 46000:
    return 'Low'
  elif income > 46000 and income < 73000:
    return 'Medium'
  else:
    return 'High'
file['Income Level'] = file['Income'].apply(determine_level)


sea.boxplot(data=file, x="Income Level", y="Addiction Level", ax=axs[0,0])
axs[0, 0].set_title("Income Level vs Addiction")



# sea.stripplot(data=file, x="Self Control", y="Addiction Level")



sea.boxplot(data=file, x="DeviceType", y="Addiction Level", ax=axs[0, 1])
axs[0, 1].set_title("Device Type vs Addiction")



sea.boxplot(data=file, x="Watch Reason", y="Addiction Level", ax=axs[0, 2])
axs[0, 2].set_title("Watch Reason vs Addiction")



def determine_age_level(age):
  if age <= 33:
    return 'Young'
  elif age > 33 and age < 50:
    return 'Middle-Age'
  else:
    return 'Older Adults'
file['Age Level'] = file['Age'].apply(determine_age_level)


sea.boxplot(data=file, x="Age Level", y="Addiction Level", ax=axs[1, 0])
axs[1, 0].set_title("Age Level vs Addiction")



sea.boxplot(data=file, x="Gender", y="Addiction Level", ax=axs[1, 1])
axs[1, 1].set_title("Gender vs Addiction")



# sea.boxplot(data=file, x="Location", y="Addiction Level", ax=axs[2, 0])



sea.boxplot(data=file, x="Debt", y="Addiction Level", ax=axs[1, 2])
axs[1, 2].set_title("Debt vs Addiction")

plt.figure()
sea.stripplot(data=file, x="ProductivityLoss", y="Addiction Level")
plt.show()

#Kmeans

filtered_df = file[['Addiction Level', 'ProductivityLoss', 'Self Control']]
data_tensor = filtered_df.to_numpy()

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=1)
kmeans.fit(data_tensor)

clusters=kmeans.cluster_centers_
print(clusters)
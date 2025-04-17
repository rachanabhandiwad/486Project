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
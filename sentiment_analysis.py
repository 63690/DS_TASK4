import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
csv_file_path = "twitter_training.csv"  # Ensure the file is in your working directory
df = pd.read_csv(csv_file_path, names=["ID", "Topic", "Sentiment", "Text"], encoding="utf-8", skiprows=1)

# Display basic dataset info
print("Dataset Overview:")
print(df.head())

# Sentiment distribution
sentiment_counts = df["Sentiment"].value_counts()

# Plot sentiment distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution in Social Media Data")
plt.xticks(rotation=45)
plt.show()

# Group data by topic and sentiment
topic_sentiment = df.groupby(["Topic", "Sentiment"]).size().unstack().fillna(0)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(topic_sentiment, annot=True, fmt=".0f", cmap="coolwarm")
plt.xlabel("Sentiment")
plt.ylabel("Topic")
plt.title("Sentiment Heatmap by Topic")
plt.show()

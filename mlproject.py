import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read data
data = pd.read_csv('training.csv')
print(data.head())

# Calculate values for new columns
data_numeric = data.drop(['class', 'max_ndvi'], axis = 1)
data['mean'] = data_numeric.mean(axis=1)
data['median'] = data_numeric.median(axis=1)
data['min_ndvi'] = data_numeric.min(axis=1)
data['std'] = data_numeric.std(axis=1)
data['range'] = data_numeric.max(axis=1) - data_numeric.min(axis=1)

# New data set with only new columns
data = data[['class', 'max_ndvi', 'min_ndvi', 'range', 'mean', 'median', 'std']]
print(data.head())

# Plotting bar graphs of means feature values by class
data.groupby('class')['mean'].mean().plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Mean Value')
plt.title('Mean Values by Class')
plt.tight_layout()
plt.show()

data.groupby('class')['median'].mean().plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Median Value')
plt.title('Median Values by Class')
plt.tight_layout()
plt.show()

data.groupby('class')[['min_ndvi', 'max_ndvi', 'range']].mean().plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Mean Value')
plt.title('Mean of Min NDVI, Max NDVI, and Range by Class')
plt.tight_layout()
plt.show()

data.groupby('class')[['std']].mean().plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Mean Value')
plt.title('Mean of standard deviation by class')
plt.tight_layout()
plt.show()


# Turn classes into integers
class_map = {
    'water': 0,
    'forest': 1,
    'impervious': 2,
    'farm': 3,
    'grass': 4,
    'orchard': 5
}

data['class_num'] = data['class'].map(class_map)

# Feature Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.drop('class', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

data = data[['class', 'max_ndvi', 'min_ndvi', 'median', 'std']]
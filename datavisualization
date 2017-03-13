import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

#Load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'class']
dataset = pd.read_csv(url, names=names)

#Data inspection
print(dataset.shape)
print(dataset.head())
print(dataset["class"].value_counts())

#Create simple scatter plot
dataset.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
plt.show()

#Create joint scatter and histo plot
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=dataset, size=7)
plt.show()

#Scatterplot with color 
sns.FacetGrid(dataset, hue="class", size=7)\
                    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
                    .add_legend()
plt.show()

#Box plot
sns.boxplot(x="class", y="SepalLengthCm", data=dataset)
plt.show()

#box plot with scatter points
ax = sns.boxplot(x="class", y="PetalLengthCm", data=dataset)
ax = sns.stripplot(x="class", y="PetalLengthCm", data=dataset, jitter=True, edgecolor="black")
plt.show()

#Violin Plot
sns.violinplot(x="class", y="PetalLengthCm", data=dataset, size=6)
plt.show()

#Kernal density estimate
sns.FacetGrid(dataset, hue="class", size=6) \
                    .map(sns.kdeplot, "SepalLengthCm") \
                    .add_legend()
plt.show()

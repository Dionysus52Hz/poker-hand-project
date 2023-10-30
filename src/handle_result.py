import pandas as pd
import matplotlib.pyplot as plt

# Create data for the table
data = {
    "Name": ["John", "Emily", "Michael", "Jessica"],
    "Age": [25, 28, 30, 23],
    "City": ["New York", "London", "Paris", "Sydney"],
}

# Create a dataframe from the data
df = pd.DataFrame(data)

# Create a table with colors
fig, ax = plt.subplots()
ax.axis("off")
table = ax.table(
    cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
)

# Set colors for the cells
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 1.2)
table._cells[(0, 0)].set_facecolor("lightgrey")  # Color for header cell
for i in range(1, len(df.index) + 1):
    for j in range(len(df.columns)):
        table._cells[(i, j)].set_facecolor("lightblue")  # Color for data cells

# Display the table
plt.show()

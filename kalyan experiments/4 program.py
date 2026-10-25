import pandas as pd

# Create data
data = {
    "Student Name": ["Ganesh", "Ravi", "Kiran", "Sita"],
    "Grade": ["A", "B", "A+", "B+"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print(df)
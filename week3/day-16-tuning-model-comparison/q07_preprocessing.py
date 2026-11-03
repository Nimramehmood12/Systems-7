import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pd.DataFrame({"age": [21, 35, 48], "income": [30_000, 55_000, 80_000],
                     "city": ["Lahore", "Karachi", "Lahore"]})
transformer = ColumnTransformer([
    ("numeric", StandardScaler(), ["age", "income"]),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
print(transformer.fit_transform(data).toarray())

# import pandas as pd

# # Cambia la ruta si es necesario
# df = pd.read_csv("past_data.csv")

# # 1.a) Asegúrate de que las columnas existan
# print("Columnas disponibles:", list(df.columns))

# # 1.b) Revisa cuántas filas cumplen Y == X
# # Por ejemplo, si X es la columna "X_value" y Y es la columna "target":
# col_X = "X"  # ajústalo al nombre de tu columna
# col_Y = "Y"  # ajústalo al nombre real de tu target
# match_mask = df[col_X] == df[col_Y]
# imatch_mask = df[col_X] != df[col_Y]
# print(f"Filas con {col_X} == {col_Y} :", match_mask.sum(), "de", len(df))

# # 1.c) Si hubo NaNs o strings, el == fallará; revisa nulos y dtypes:
# print("Valores nulos:\n", df[[col_X, col_Y]].isnull().sum())
# print("Tipos de datos:\n", df[[col_X, col_Y]].dtypes)
# print("Valores distintos en", col_X, ":", df[imatch_mask])


a = []

print(a[-1])

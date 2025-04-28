import pandas as pd

# Cargar los datasets desde archivos Excel
df1 = pd.read_excel('pruebafinal_reviews_rimac.xlsx')
df2 = pd.read_excel('pruebafinal_reviews_mapfre.xlsx')
df3 = pd.read_excel('pruebafinal_reviews_pacifico.xlsx')

# Unirlos en uno solo
df_combinado = pd.concat([df1, df2, df3], ignore_index=True)

# Guardar el dataset combinado en un nuevo archivo Excel
df_combinado.to_excel('reviews_test.xlsx', index=False)

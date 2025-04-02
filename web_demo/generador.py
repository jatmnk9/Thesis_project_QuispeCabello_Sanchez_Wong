import pandas as pd

# Cargar el archivo Excel
ruta_excel = "sentiment_analysis_results.xlsx"  # Asegúrate de poner el nombre correcto
df = pd.read_excel(ruta_excel, sheet_name="Ranking_Mensual")

# Convertir a HTML (sin índices)
tabla_html = df.to_html(index=False, classes="table table-striped")
print(tabla_html)  # Copia este HTML en tu código

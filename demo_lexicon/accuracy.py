import pandas as pd

# 1. Cargar datasets
df_manual = pd.read_excel("reseñas_emociones_etiquetadas.xlsx")
df_lexicon = pd.read_excel("REVISAR.xlsx")

# 2. Normalizar columna 'review' a minúsculas
df_manual['review'] = df_manual['review'].astype(str).str.lower()
df_lexicon['review'] = df_lexicon['review'].astype(str).str.lower()

# 3. Mapear etiquetas manuales (i, s, n → texto completo)
etiqueta_map = {'i': 'Negativo', 's': 'Positivo', 'n': 'Neutro'}
df_manual['sentiment_manual'] = df_manual['etiqueta'].map(etiqueta_map)

# 4. Renombrar columna de sentimiento automático si es necesario
df_lexicon.rename(columns={'sentiment': 'sentiment_lexicon'}, inplace=True)

# 5. Eliminar columnas duplicadas por si acaso
df_manual = df_manual.loc[:, ~df_manual.columns.duplicated()]
df_lexicon = df_lexicon.loc[:, ~df_lexicon.columns.duplicated()]

# 6. Hacer el merge por 'review'
df_comparado = pd.merge(
    df_lexicon[['review', 'sentiment_lexicon']].drop_duplicates(),
    df_manual[['review', 'sentiment_manual']].drop_duplicates(),
    on='review',
    how='inner'
)

# 7. Crear columna de comparación
df_comparado['comparacion'] = df_comparado.apply(
    lambda row: 'COINCIDE' if row['sentiment_lexicon'] == row['sentiment_manual'] else 'NO COINCIDE',
    axis=1
)

# 8. Calcular exactitud
exactitud = (df_comparado['comparacion'] == 'COINCIDE').mean()
print(f"✔️ Exactitud del modelo basado en nrc: {exactitud:.2%}")

# 9. Exportar resultados
df_comparado.to_excel("comparacion_sentimientos.xlsx", index=False)

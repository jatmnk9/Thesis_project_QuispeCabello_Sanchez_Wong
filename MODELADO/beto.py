import pandas as pd
from transformers import pipeline  # Usamos la API simplificada
from tqdm import tqdm
from sklearn.metrics import classification_report

# Configuración inicial
tqdm.pandas()

# 1. Cargar BETO usando pipeline (evita problemas con TensorFlow)
analyzer = pipeline(
    "text-classification",
    model="finiteautomata/beto-sentiment-analysis",
    tokenizer="finiteautomata/beto-sentiment-analysis"
)

# 2. Función optimizada para clasificación
def clasificar_con_beto(texto):
    result = analyzer(texto[:512])  # Limitamos a 512 tokens (máximo de BERT)
    return result[0]['label'].lower()  # Convertimos a minúsculas

# 3. Cargar datos
df_reviews = pd.read_excel("reviews_test.xlsx")

# 4. Preprocesamiento mínimo
df_reviews['review_clean'] = df_reviews['review'].astype(str).str.strip()

# 5. Aplicar BETO (con barra de progreso)
print("\n🔍 Analizando reseñas con BETO...")
df_reviews['prediccion_beto'] = df_reviews['review_clean'].progress_apply(clasificar_con_beto)

# 6. Mapeo de etiquetas (opcional, según el modelo)
label_map = {
    'pos': 'positivo',
    'neg': 'negativo',
    'neu': 'neutro'
}
df_reviews['prediccion_beto'] = df_reviews['prediccion_beto'].map(label_map).fillna(df_reviews['prediccion_beto'])

# 7. Evaluación (si hay etiquetas)
if 'etiqueta' in df_reviews.columns:
    df_reviews['etiqueta_limpia'] = df_reviews['etiqueta'].str.strip().str.lower()
    print("\n📊 Resultados de BETO:")
    print(classification_report(
        df_reviews['etiqueta_limpia'],
        df_reviews['prediccion_beto'],
        target_names=['negativo', 'neutro', 'positivo']
    ))

# 8. Exportar resultados
output_cols = ['review', 'review_clean', 'prediccion_beto']
if 'etiqueta' in df_reviews.columns:
    output_cols.append('etiqueta')

df_reviews[output_cols].to_excel("Resultados_BETO.xlsx", index=False)
print("\n✅ Análisis completado. Resultados guardados en 'Resultados_BETO.xlsx'")
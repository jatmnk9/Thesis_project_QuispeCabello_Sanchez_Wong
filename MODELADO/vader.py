import pandas as pd
import re
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report

# Configuración inicial
tqdm.pandas()

# Cargar datos
df_reviews = pd.read_excel("reviews_test.xlsx")

# Preprocesamiento mínimo para VADER (solo limpia URLs y menciones)
def preprocesar_texto(texto):
    texto = str(texto)
    # Eliminar URLs, menciones y hashtags (opcional)
    texto = re.sub(r'http\S+|@\w+|#\w+', '', texto)
    return texto.strip()

df_reviews['review_clean'] = df_reviews['review'].progress_apply(preprocesar_texto)

# Inicializar VADER
analyzer = SentimentIntensityAnalyzer()

# Función para análisis de sentimiento con VADER
def analyze_with_vader(text):
    scores = analyzer.polarity_scores(text)
    return pd.Series([
        scores['neg'],  # Negatividad
        scores['neu'],  # Neutralidad
        scores['pos'],  # Positividad
        scores['compound']  # Puntuación compuesta (-1 a 1)
    ])

# Aplicar VADER
df_reviews[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = \
    df_reviews['review_clean'].progress_apply(analyze_with_vader)

# Clasificación basada en el compound score de VADER
def clasificar_con_vader(row):
    if row['vader_compound'] > 0.05:
        return 'positivo'
    elif row['vader_compound'] < -0.05:
        return 'negativo'
    else:
        return 'neutro'

df_reviews['prediccion_vader'] = df_reviews.apply(clasificar_con_vader, axis=1)

# (Opcional) Comparar con etiquetas reales si existen
if 'etiqueta' in df_reviews.columns:
    df_reviews['etiqueta_limpia'] = df_reviews['etiqueta'].str.strip().str.lower()
    print("\n🔍 Resultados de VADER:")
    print(classification_report(
        df_reviews['etiqueta_limpia'], 
        df_reviews['prediccion_vader'],
        target_names=['negativo', 'neutro', 'positivo']
    ))

# Exportar resultados
output_columns = [
    'review',
    'review_clean',
    'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
    'prediccion_vader'
]

# Agregar etiqueta si existe
if 'etiqueta' in df_reviews.columns:
    output_columns.append('etiqueta')

df_reviews[output_columns].to_excel("Resultados_VADER.xlsx", index=False)

print("\n✅ Análisis con VADER completado. Revisa 'Resultados_VADER.xlsx'")
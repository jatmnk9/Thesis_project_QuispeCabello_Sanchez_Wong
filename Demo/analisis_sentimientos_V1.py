import pandas as pd
import numpy as np
import re
import unidecode
from collections import defaultdict
from nrclex import NRCLex
from deep_translator import GoogleTranslator
from tqdm import tqdm

# Cargar el dataset
reviews_df = pd.read_excel("comisarias_reviews.xlsx")

# Asegurar que la columna de fecha ya está en formato correcto
reviews_df['review_date'] = reviews_df['review_date'].astype(str)

# Filtrar fechas en formato MM/YY
valid_date_pattern = re.compile(r'^(0[1-9]|1[0-2])\/\d{2}$')
reviews_df = reviews_df[reviews_df['review_date'].apply(lambda x: bool(valid_date_pattern.match(x)))]

# Función para traducir las reseñas
translator = GoogleTranslator(source='es', target='en')
def translate_text(text):
    try:
        return translator.translate(str(text))
    except Exception as e:
        return ""

# Aplicar traducción con barra de progreso
tqdm.pandas()
reviews_df['translated_review'] = reviews_df['review'].progress_apply(translate_text)

# Función para obtener las emociones con NRCLex
def get_sentiment(text):
    text = unidecode.unidecode(str(text).lower().strip())
    text_object = NRCLex(text)
    scores = text_object.raw_emotion_scores
    return scores

# Aplicar análisis de sentimiento con barra de progreso
reviews_df['sentiment_scores'] = reviews_df['translated_review'].progress_apply(get_sentiment)

# Calcular polaridad y puntajes de emoción
reviews_df['polarity_score'] = reviews_df['sentiment_scores'].apply(lambda x: x.get('positive', 0) - x.get('negative', 0))
reviews_df['weighted_score'] = reviews_df['polarity_score'] * reviews_df['rating']

# Crear rankings históricos, anuales y mensuales
reviews_df['year'] = reviews_df['review_date'].str[-2:].astype(str)
reviews_df['year_month'] = reviews_df['review_date']

ranking_historico = reviews_df.groupby('place')['weighted_score'].mean().sort_values(ascending=False).reset_index()
ranking_anual = reviews_df.groupby(['year', 'place'])['weighted_score'].mean().reset_index()
ranking_mensual = reviews_df.groupby(['year_month', 'place'])['weighted_score'].mean().reset_index()

# Guardar resultados en un Excel
with pd.ExcelWriter("sentiment_analysis_results.xlsx") as writer:
    reviews_df.to_excel(writer, sheet_name="Reviews", index=False)
    ranking_historico.to_excel(writer, sheet_name="Ranking_Historico", index=False)
    ranking_anual.to_excel(writer, sheet_name="Ranking_Anual", index=False)
    ranking_mensual.to_excel(writer, sheet_name="Ranking_Mensual", index=False)

print("Análisis de sentimiento completado con NRCLex. Resultados guardados en 'sentiment_analysis_results.xlsx'")

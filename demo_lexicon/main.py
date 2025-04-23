import pandas as pd
import re
from tqdm import tqdm

# Activar tqdm para pandas
tqdm.pandas()

# 1. Cargar datasets
df_reviews = pd.read_excel("reviews.xlsx")
df_lexicon = pd.read_excel("Lexicon.xlsx", sheet_name="lexiconv2_lematizado")

# 2. Limpiar columnas necesarias
df_reviews['review_clean'] = df_reviews['review_clean'].astype(str).str.lower()
df_lexicon['Text'] = df_lexicon['Text'].astype(str).str.lower()

# 3. Convertir el lexicón a lista de patrones regex
lexicon_entries = []
for _, row in df_lexicon.iterrows():
    phrase = row['Text'].strip()
    if ' ' in phrase:
        pattern = re.escape(phrase)  # frase exacta
    else:
        pattern = r'\b' + re.escape(phrase) + r'\b'  # palabra exacta
    lexicon_entries.append({
        'pattern': pattern,
        'Positive': row['Positive'],
        'Negative': row['Negative']
    })

# 4. Función de análisis con regex
def analyze_sentiment_regex(review):
    pos_count, neg_count = 0, 0
    for entry in lexicon_entries:
        matches = re.findall(entry['pattern'], review)
        count = len(matches)
        if entry['Positive'] == 1:
            pos_count += count
        if entry['Negative'] == 1:
            neg_count += count
    
    if pos_count > neg_count:
        sentiment = 'Positivo'
    elif neg_count > pos_count:
        sentiment = 'Negativo'
    else:
        sentiment = 'Neutro'
        
    return pd.Series([pos_count, neg_count, sentiment])

# 5. Aplicar análisis con barra de progreso
df_reviews[['pos_count', 'neg_count', 'sentiment']] = df_reviews['review_clean'].progress_apply(analyze_sentiment_regex)

# 6. Exportar resultados
df_reviews.to_excel("sentimiento_por_lexicon.xlsx", index=False)

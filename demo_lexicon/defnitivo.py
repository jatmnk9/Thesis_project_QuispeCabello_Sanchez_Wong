import pandas as pd
import re
import unicodedata
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk

# Preparación
tqdm.pandas()
nltk.download('stopwords')
stopwords_es = set(stopwords.words('spanish'))

# Cargar modelo SpaCy
nlp = spacy.load("es_core_news_sm")

# Modificadores y verbos auxiliares
modificadores = {'no', 'nunca', 'jamás', 'nada', 'nadie', 'ninguno', 'ni', 'poco', 'muy', 'tan', 'bastante', 'demasiado', 'otro', 'mucho'}
verbos_auxiliares = {'ser', 'estar', 'haber'}

# Función para quitar tildes
def quitar_tildes(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

# Función de limpieza y lematización
def limpiar_texto(texto):
    texto = quitar_tildes(str(texto).lower())
    doc = nlp(texto)
    tokens = [
        token.lemma_ 
        for token in doc 
        if token.is_alpha and (token.lemma_ not in stopwords_es or token.lemma_ in modificadores)
        and token.lemma_ not in verbos_auxiliares
    ]
    return ' '.join(tokens)

# Cargar datos
df_reviews = pd.read_excel("reseñas_etiquetadas.xlsx")
df_lexicon = pd.read_excel("lexicon.xlsx", sheet_name="lexiconv2_lematizado")

# Preprocesar léxico y reseñas
df_reviews['review_clean'] = df_reviews['review'].progress_apply(limpiar_texto)
df_lexicon['Text'] = df_lexicon['Text'].astype(str).str.lower()
df_lexicon['pattern'] = df_lexicon['Text'].apply(lambda x: re.escape(x) if ' ' in x else r'\b' + re.escape(x) + r'\b')
lexicon_entries = df_lexicon.to_dict(orient='records')

# Función de análisis por frases del léxico
def analyze_sentiment_phrases(text):
    matched_phrases = []
    text_copy = text

    for entry in lexicon_entries:
        matches = list(re.finditer(entry['pattern'], text_copy))
        for match in matches:
            matched_phrases.append(entry['Text'])
            start, end = match.span()
            text_copy = text_copy[:start] + ' ' * (end - start) + text_copy[end:]

    tokens_positivos = [t for t in matched_phrases if df_lexicon[df_lexicon['Text'] == t]['Positive'].values[0] == 1]
    tokens_negativos = [t for t in matched_phrases if df_lexicon[df_lexicon['Text'] == t]['Negative'].values[0] == 1]

    pos_count = len(tokens_positivos)
    neg_count = len(tokens_negativos)

    if pos_count > neg_count:
        sentiment = 'satisfecho'
    elif neg_count > pos_count:
        sentiment = 'insatisfecho'
    else:
        sentiment = 'neutro'

    return pd.Series([pos_count, neg_count, sentiment, matched_phrases, tokens_positivos, tokens_negativos])

# Aplicar análisis
df_reviews[['pos_count', 'neg_count', 'sent_pred', 'tokens_encontrados', 'tokens_positivos', 'tokens_negativos']] = \
    df_reviews['review_clean'].progress_apply(analyze_sentiment_phrases)

# Comparar con etiquetas reales
df_reviews['match'] = df_reviews.apply(lambda x: x['etiqueta'].strip().lower() == x['sent_pred'], axis=1)

# Métricas
accuracy = df_reviews['match'].mean() * 100
por_tipo = df_reviews.groupby('etiqueta')['match'].mean() * 100

print(f"Coincidencia general: {accuracy:.2f}%")
print("Coincidencias por etiqueta:")
print(por_tipo)

# Exportar resultados
df_reviews.to_excel("reseñas_clasificadas_con_tokens.xlsx", index=False)
print("✅ Análisis completado. Revisa 'reseñas_clasificadas_con_tokens.xlsx'")
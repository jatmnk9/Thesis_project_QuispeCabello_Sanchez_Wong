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
df_reviews = pd.read_excel("reviews_test.xlsx")
df_lexicon = pd.read_excel("Lexicon.xlsx", sheet_name="lexiconv2_lematizado")

# Preprocesar léxico y reseñas
df_reviews['review_clean'] = df_reviews['review'].progress_apply(limpiar_texto)
df_lexicon['Text'] = df_lexicon['Text'].astype(str).str.lower()
df_lexicon['pattern'] = df_lexicon['Text'].apply(lambda x: re.escape(x) if ' ' in x else r'\b' + re.escape(x) + r'\b')
lexicon_entries = df_lexicon.to_dict(orient='records')



def analyze_sentiment_phrases(text):
    matched_phrases = []
    text_lower = text.lower()
    text_mask = ['0'] * len(text_lower)  # Para marcar posiciones ya ocupadas

    # Ordenar las frases del léxico por cantidad de palabras (de más a menos)
    lexicon_sorted = sorted(lexicon_entries, key=lambda x: len(x['Text'].split()), reverse=True)

    for entry in lexicon_sorted:
        phrase = entry['Text'].lower()
        pattern = re.escape(phrase)
        regex_pattern = r'\b' + pattern + r'\b'
        
        for match in re.finditer(regex_pattern, text_lower):
            start, end = match.span()
            # Verificamos que esta parte del texto no esté ya ocupada por otra coincidencia
            if all(c == '0' for c in text_mask[start:end]):
                matched_phrases.append(entry['Text'])  # Guarda la versión original
                for i in range(start, end):
                    text_mask[i] = '1'  # Marca como ocupada

    # Extraer positivos y negativos según df_lexicon
    tokens_positivos = [t for t in matched_phrases if df_lexicon[df_lexicon['Text'] == t]['Positive'].values[0] == 1]
    tokens_negativos = [t for t in matched_phrases if df_lexicon[df_lexicon['Text'] == t]['Negative'].values[0] == 1]

    pos_count = len(tokens_positivos)
    neg_count = len(tokens_negativos)

    if pos_count > neg_count:
        sentiment = 'Positivo'
    elif neg_count > pos_count:
        sentiment = 'Negativo'
    else:
        sentiment = 'Neutro'

    return pd.Series([pos_count, neg_count, sentiment, matched_phrases, tokens_positivos, tokens_negativos])


# Aplicar análisis
df_reviews[['pos_count', 'neg_count', 'sent_pred', 'tokens_encontrados', 'tokens_positivos', 'tokens_negativos']] = \
    df_reviews['review_clean'].progress_apply(analyze_sentiment_phrases)

# Asegurarse de que las columnas estén en el formato correcto (sin espacios, en minúsculas)
df_reviews['etiqueta'] = df_reviews['etiqueta'].str.strip().str.lower()
df_reviews['sent_pred'] = df_reviews['sent_pred'].str.strip().str.lower()

# Comparar predicción con la etiqueta real
df_reviews['match'] = df_reviews['etiqueta'] == df_reviews['sent_pred']

# Calcular precisión general
accuracy = df_reviews['match'].mean() * 100

# Calcular precisión por tipo de etiqueta
por_tipo = df_reviews.groupby('etiqueta')['match'].mean() * 100

# Mostrar resultados
print(f"Precisión general: {accuracy:.2f}%")
print("Precisión por tipo de etiqueta:")
print(por_tipo)

# Exportar resultados
df_reviews.to_excel("REVISAR.xlsx", index=False)
print("✅ Análisis completado. Revisa 'REVISAR.xlsx'")
import pandas as pd
import re
from collections import defaultdict, Counter
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from tqdm import tqdm

# Cargar modelo de lenguaje en español
nlp = spacy.load("es_core_news_sm")

# Cargar el dataset
reviews_df = pd.read_excel("comisarias_reviews.xlsx")

# Asegurar que la columna de fecha ya está en formato correcto
reviews_df['review_date'] = reviews_df['review_date'].astype(str)

# Filtrar fechas en formato MM/YY
valid_date_pattern = re.compile(r'^(0[1-9]|1[0-2])\/\d{2}$')
reviews_df = reviews_df[reviews_df['review_date'].apply(lambda x: bool(valid_date_pattern.match(x)))]

# Extraer año y mes para rankings
reviews_df['year'] = reviews_df['review_date'].apply(lambda x: '20' + x.split('/')[1])
reviews_df['month'] = reviews_df['review_date'].apply(lambda x: x.split('/')[0])

# Función para calcular ranking
def calculate_ranking(df, group_by):
    ranking = df.groupby(group_by).agg(
        avg_rating=('rating', 'mean'),
        review_count=('rating', 'count')
    ).reset_index()
    ranking = ranking.sort_values(by=['avg_rating', 'review_count'], ascending=[False, False])
    ranking['rank'] = range(1, len(ranking) + 1)
    return ranking

# Rankings
historical_ranking = calculate_ranking(reviews_df, ['place', 'country', 'city', 'address'])
annual_ranking = calculate_ranking(reviews_df, ['place', 'country', 'city', 'address', 'year'])
monthly_ranking = calculate_ranking(reviews_df, ['place', 'country', 'city', 'address', 'year', 'month'])

# Guardar rankings en un Excel
with pd.ExcelWriter("ranking_reviews.xlsx") as writer:
    historical_ranking.to_excel(writer, sheet_name='Historical Ranking', index=False)
    annual_ranking.to_excel(writer, sheet_name='Annual Ranking', index=False)
    monthly_ranking.to_excel(writer, sheet_name='Monthly Ranking', index=False)

# Función para extraer categorías más mencionadas
def extract_keywords(text, top_n=5):
    doc = nlp(text.lower())
    words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    most_common = Counter(words).most_common(top_n)
    return ', '.join([word for word, _ in most_common])

# Función para generar resúmenes de reseñas
def generate_summary(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("spanish"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return '\n'.join(str(sentence) for sentence in summary)

# Generar y guardar resúmenes con barra de progreso
places_reviews = defaultdict(str)
for _, row in reviews_df.iterrows():
    review_text = str(row['review']) if pd.notna(row['review']) else ""
    places_reviews[row['place']] += review_text + " "

print("Generando resúmenes...")
for place, reviews in tqdm(places_reviews.items(), desc="Procesando lugares"):
    summary = generate_summary(reviews)
    keywords = extract_keywords(reviews)
    with open(f"{place.replace(' ', '_')}_summary.txt", "w", encoding="utf-8") as file:
        file.write(f"Resumen:\n{summary}\n\nCategorías más mencionadas: {keywords}")

print("Rankings generados y resúmenes guardados correctamente.")
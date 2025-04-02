import re
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_google_maps_reviews(place_url):
    """
    Scrapes Google Maps reviews from a direct URL and returns the data in a DataFrame.
    """

    service = Service()
    options = webdriver.ChromeOptions()
    #options.add_argument('--headless')  # Run Chrome in headless mode
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(place_url)

    try:
        # Esperar a que cargue la página de la comisaría
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "DUwDvf")))

        # Obtener el nombre y la dirección
        place_name_element = driver.find_element(By.CLASS_NAME, "DUwDvf")
        place_name_text = place_name_element.text.strip()
        
        address_element = driver.find_element(By.CLASS_NAME, "Io6YTe")
        address_text = address_element.text.strip()

        # Hacer clic en la pestaña de reseñas
        try:
            reviews_tab = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Opiniones')]"))
            )
            reviews_tab.click()
        except Exception as e:
            print(f"Error al hacer clic en la pestaña de reseñas: {e}")
            driver.quit()
            return pd.DataFrame()  # Retorna un DataFrame vacío si falla

        # Esperar a que aparezca la sección de reseñas
        reviews_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "m6QErb"))
        )

                # Esperar a que el contenedor de reseñas esté presente
        scrollable_div = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde"))
        )

        last_review_count = 0
        scroll_attempts = 0
        max_attempts = 2

        while scroll_attempts < max_attempts:
            # Hacer scroll dentro del contenedor específico de reseñas
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
            time.sleep(2)

            # Contar cuántas reseñas hay cargadas hasta ahora
            soup = BeautifulSoup(driver.page_source, "html.parser")
            reviews_elements = soup.find_all("div", class_="jftiEf")
            current_review_count = len(reviews_elements)

            print(f"🔍 Reseñas cargadas: {current_review_count}")

            if current_review_count == last_review_count:
                scroll_attempts += 1
                print(f"⚠️ No nuevas reseñas. Intento {scroll_attempts}/{max_attempts}")
            else:
                last_review_count = current_review_count
                scroll_attempts = 0  # Resetear si hay nuevas reseñas

        # Parsear las reseñas
        soup = BeautifulSoup(driver.page_source, "html.parser")
        reviews_elements = soup.find_all("div", class_="jftiEf")

        reviews_data = []
        for review_element in reviews_elements:
            try:
                author_element = review_element.find("div", class_="d4r55")
                author = author_element.text.strip() if author_element else "N/A"

                review_text_element = review_element.find("span", class_="wiI7pd")
                review_text = review_text_element.text.strip() if review_text_element else "N/A"

                raw_rating = review_element.find("span", class_="kvMYJc")["aria-label"]
                rating = float(re.search(r"(\d+)", raw_rating).group(1))

                review_date_element = review_element.find("span", class_="rsqaWe")
                review_date = review_date_element.text.strip() if review_text_element else "N/A"

                  
                reviews_data.append({
                    "place": place_name_text,
                    "country": "Perú",
                    "departamento": "Lima",
                    "provincia": "Lima",
                    "distrito": "Lima",
                    "dirección": address_text,
                    "autor": author,
                    "reseña": review_text,
                    "estrellas": rating,
                    "fecha": review_date
                })
            except Exception as e:
                print(f"Error procesando una reseña: {e}")
                
    except Exception as e:
        print(f"Error obteniendo reseñas de {place_url}: {e}")
        reviews_data = []

    driver.quit()
    return pd.DataFrame(reviews_data)

def scrape_reviews(category, urls):
    all_reviews = pd.DataFrame()
    for url in urls:
        reviews_df = get_google_maps_reviews(url)
        all_reviews = pd.concat([all_reviews, reviews_df], ignore_index=True)
    
    if not all_reviews.empty:
        filename = f"{category}_reviews.xlsx"
        all_reviews.to_excel(filename, index=False)
        print(f"✅ Reseñas guardadas en {filename}")
    else:
        print(f"⚠️ No se encontraron reseñas para {category}.")

if __name__ == "__main__":
    categories = {
        "comisarias": [
            "https://www.google.com/maps/search/Comisaria+Mirones+Alto",
            "https://www.google.com/maps/search/Comisaría+Salamanca+-+Ate",
            "https://www.google.com/maps/search/Comisaria+Pnp+Unidad+Vecinal+De+Mirones"
        ],
        "ministerios": [
            "https://www.google.com/maps/search/Ministerio+de+Relaciones+Exteriores+del+Perú+(RREE)",
            "https://www.google.com/maps/search/Ministerio+de+Educación+del+Perú",
            "https://www.google.com/maps/search/Ministerio+de+Justicia+y+Derechos+Humanos+del+Perú+(MINJUS)"
            "https://www.google.com/maps/search/Ministerio+del+Interior+del+Perú+(MININTER)",
            "https://www.google.com/maps/search/Ministerio+de+la+Mujer+y+Poblaciones+Vulnerables"
        ],
        "cines": [
            "https://www.google.com/maps/search/Cineplanet+Centro+Cívico",
            "https://www.google.com/maps/search/Cineplanet+Plaza+San+Miguel",
            "https://www.google.com/maps/search/Cineplanet+Brasil",
            "https://www.google.com/maps/search/Cineplanet+Salaverry",
            "https://www.google.com/maps/search/Cinestar+UNI",
            "https://www.google.com/maps/search/CineStar+Benavides",
            "https://www.google.com/maps/search/CINESTAR+SAN+JUAN",
            "https://www.google.com/maps/search/UVK+Platino+Panorama",
            "https://www.google.com/maps/search/UVK+Multicines+El+Agustino"
        ]
    }
    
    for category, urls in categories.items():
        scrape_reviews(category, urls)

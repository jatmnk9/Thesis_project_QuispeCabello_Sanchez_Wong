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

        # Hacer scroll dentro del contenedor de reseñas
        scroll_attempts = 0
        while scroll_attempts < 30:  # Intenta hacer scroll hasta que deje de cargar reseñas
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", reviews_container)
            time.sleep(1.5)  # Espera un poco para que carguen más reseñas
            scroll_attempts += 1

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

                reviews_data.append({
                    "nombre comisaría": place_name_text,
                    "país": "Perú",
                    "departamento": "Lima",
                    "provincia": "Lima",
                    "distrito": address_text.split(",")[1].strip().split()[0], # Extrae el distrito
                    "dirección": address_text,
                    "autor": author,
                    "reseña": review_text,
                    "estrellas": rating,
                })
            except Exception as e:
                print(f"Error procesando una reseña: {e}")
                
    except Exception as e:
        print(f"Error obteniendo reseñas de {place_url}: {e}")
        reviews_data = []

    driver.quit()
    return pd.DataFrame(reviews_data)

if __name__ == "__main__":
    comisaria_urls = [
        "https://www.google.com/maps/search/Comisaria+Mirones+Alto",
        "https://www.google.com/maps/search/Comisaría+Salamanca+-+Ate"
    ]

    all_reviews = pd.DataFrame()
    for url in comisaria_urls:
        reviews_df = get_google_maps_reviews(url)
        all_reviews = pd.concat([all_reviews, reviews_df], ignore_index=True)

    if not all_reviews.empty:
        all_reviews.to_excel("comisarias_reviews.xlsx", index=False)
        print("✅ Reseñas guardadas en comisarias_reviews.xlsx")
    else:
        print("⚠️ No se encontraron reseñas.")

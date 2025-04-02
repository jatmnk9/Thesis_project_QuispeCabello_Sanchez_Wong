import time
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class GoogleMapsScraper:
    def __init__(self):
        """Inicializa el driver de Selenium con opciones optimizadas."""
        service = Service()
        options = webdriver.ChromeOptions()
        #options.add_argument('--headless=new')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        self.driver = webdriver.Chrome(service=service, options=options)

    def get_reviews(self, place_url):
        """Obtiene todas las reseñas de un lugar en Google Maps."""
        self.driver.get(place_url)

        try:
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "DUwDvf")))
            place_name = self.driver.find_element(By.CLASS_NAME, "DUwDvf").text.strip()
            address = self.driver.find_element(By.CLASS_NAME, "Io6YTe").text.strip()

            try:
                reviews_tab = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "LRkQ2"))
                )
                reviews_tab.click()
                print(f"✅ {place_name}: Se hizo clic en la pestaña de reseñas correctamente.")
            except Exception as e:
                print(f"❌ Error al hacer clic en la pestaña de reseñas: {e}")
                return pd.DataFrame()

            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "jftiEf")))

            self.__scroll()
            self.__expand_reviews()
            return self.__parse_reviews(place_name, address)

        except Exception as e:
            print(f"❌ Error al obtener reseñas de {place_url}: {e}")
            return pd.DataFrame()

    def __scroll(self):
        """Hace scroll en la sección de reseñas para cargar más comentarios."""
        try:
            scrollable_div = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde"))
            )

            last_count = 0
            scroll_attempts = 0

            while scroll_attempts < 10:
                self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
                time.sleep(2)

                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                reviews_elements = soup.find_all("div", class_="jftiEf")
                new_count = len(reviews_elements)

                if new_count > last_count:
                    last_count = new_count
                    scroll_attempts = 0
                else:
                    scroll_attempts += 1
                    print(f"🔻 Intento {scroll_attempts}: No se detectaron nuevas reseñas.")
                    if scroll_attempts >= 5:
                        print(f"⚠️ Deteniendo scroll después de {scroll_attempts} intentos sin nuevas reseñas.")
                        break

        except Exception as e:
            print(f"⚠️ Error al hacer scroll: {e}")

    def __expand_reviews(self):
        """Expande todas las reseñas largas que tienen un botón 'Leer más'."""
        try:
            while True:
                more_buttons = self.driver.find_elements(By.CLASS_NAME, "w8nwRe")
                if not more_buttons:
                    break
                for button in more_buttons:
                    try:
                        button.click()
                        time.sleep(0.5)
                    except:
                        pass
        except Exception as e:
            print(f"⚠️ Error al expandir reseñas: {e}")

    def __parse_reviews(self, place_name, address):
        """Extrae información detallada de cada reseña."""
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        reviews_elements = soup.find_all("div", class_="jftiEf")

        reviews_data = []
        for review in reviews_elements:
            try:
                author = review.find("div", class_="d4r55").text.strip()
                review_text_element = review.find("span", class_="wiI7pd")
                review_text = review_text_element.text.strip() if review_text_element else "N/A"

                raw_rating = review.find("span", class_="kvMYJc")["aria-label"]
                rating = float(re.search(r"(\d+)", raw_rating).group(1))

                reviews_data.append({
                    "nombre comisaría": place_name,
                    "país": "Perú",
                    "departamento": "Lima",
                    "provincia": "Lima",
                    "distrito": address.split(",")[1].strip().split()[0],
                    "dirección": address,
                    "autor": author,
                    "reseña": review_text,
                    "rating": rating,
                    "retrieval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                print(f"⚠️ Error al procesar una reseña: {e}")

        print(f"✅ {len(reviews_data)} reseñas extraídas de {place_name}")
        return pd.DataFrame(reviews_data)

    def close(self):
        """Cierra el navegador."""
        self.driver.quit()

if __name__ == "__main__":
    scraper = GoogleMapsScraper()

    comisaria_urls = [
       "https://www.google.com/maps/search/Comisaria+Mirones+Alto",
    "https://www.google.com/maps/search/Comisaría+Salamanca+-+Ate"
    ]

    all_reviews = pd.DataFrame()
    for url in comisaria_urls:
        reviews_df = scraper.get_reviews(url)
        all_reviews = pd.concat([all_reviews, reviews_df], ignore_index=True)

    scraper.close()

    if not all_reviews.empty:
        all_reviews.to_excel("comisarias_reviews.xlsx", index=False)
        print("✅ Reseñas guardadas en comisarias_reviews.xlsx")
    else:
        print("❌ No se encontraron reseñas.")
import os
import requests
import time

def download_people_photos(target_folder, count=100):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Папка '{target_folder}' створена.")

    # Використовуємо Lorem Flickr як стабільну альтернативу
    # Теги: 'face,person'
    base_url = "https://loremflickr.com/600/600/face,person/all"
    
    downloaded = 0
    while downloaded < count:
        try:
            # Параметр 'lock' змушує сервіс видавати різні фото
            response = requests.get(f"{base_url}?lock={downloaded}", timeout=15)
            
            if response.status_code == 200:
                file_path = os.path.join(target_folder, f"person_{downloaded + 1}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded += 1
                print(f"[{downloaded}/{count}] Завантажено успішно.")
                
                # Невелика пауза, щоб уникнути блокування
                time.sleep(0.2)
            else:
                print(f"Помилка {response.status_code}. Очікування 2 секунди...")
                time.sleep(2)
        
        except Exception as e:
            print(f"Помилка з'єднання: {e}")
            time.sleep(3)

    print(f"\nГотово! Всі {count} фото у папці '{target_folder}'.")

if __name__ == "__main__":
    download_people_photos("images", 100)
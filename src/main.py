import sys
import os
from dotenv import load_dotenv

# добавление корневой директории проекта в путь Python, чтобы импортировать модули из других частей проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# загрузка переменных окружения из файла .env, если он существует
load_dotenv()

# импрот приложения FastAPI
from recommendation_system.recommendation_system import app

# запуск сервера Uvicorn на локальном хосте (если скрипт запущен напрямую)
if __name__ == "__main__":
    import uvicorn
    # Получение значений переменных окружения
    host = os.getenv("HOST", "127.0.0.1")  # Значение по умолчанию "127.0.0.1"
    port = int(os.getenv("PORT", "8000"))  # Значение по умолчанию 8000
    uvicorn.run(app, host=host, port=port)
import sys
import os

# добавление корневой директории проекта в путь Python, чтобы импортировать модули из других частей проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# импрот приложения FastAPI
from recommendation_system.recomendation_system_with_vectors import app

# запуск сервера Uvicorn на локальном хосте (если скрипт запущен напрямую)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
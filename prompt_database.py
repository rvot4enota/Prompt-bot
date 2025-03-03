import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Dict, Any


class PromptDatabase:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", persist_directory="./chroma_db"):
        """
        Инициализирует базу данных промптов с использованием LangChain и Chroma

        Args:
            embedding_model_name: Название модели для эмбеддингов
            persist_directory: Директория для хранения базы данных
        """
        # Создаем директорию для хранения, если её нет
        os.makedirs(persist_directory, exist_ok=True)

        # Инициализация эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Инициализация или загрузка векторной базы данных
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Загружена существующая база данных из {persist_directory}")
        else:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Создана новая база данных в {persist_directory}")

    def load_prompts_from_dataframe(self, df: pd.DataFrame) -> int:
        """
        Загружает промпты из pandas DataFrame в базу данных

        Args:
            df: DataFrame с промптами

        Returns:
            int: Количество загруженных промптов
        """
        if df.empty:
            print("Датафрейм пуст, нет промптов для загрузки")
            return 0

        # Проверяем наличие необходимых колонок
        required_cols = ['act', 'prompt']
        for col in required_cols:
            if col not in df.columns:
                print(f"Отсутствует обязательная колонка: {col}")
                return 0

        # Преобразование данных в формат Document для LangChain
        documents = []
        for _, row in df.iterrows():
            prompt_text = row['prompt']
            # Проверка метаданных
            metadata = {
                "category": row['act'],
                "for_devs": row.get('for_devs', False)
            }
            documents.append(Document(page_content=prompt_text, metadata=metadata))

        # Добавление документов в базу данных
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()  # Сохраняем изменения

        print(f"Добавлено {len(documents)} промптов в базу данных")
        return len(documents)

    def search_prompts(self, query: str, filter_dict: dict = None, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Ищет наиболее релевантные промпты по запросу

        Args:
            query: Текст запроса для поиска
            filter_dict: Фильтр для метаданных (например, {"for_devs": True})
            n_results: Количество результатов для возврата

        Returns:
            List[Dict]: Список найденных промптов с метаданными
        """
        # Поиск похожих документов с помощью LangChain
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=n_results,
            filter=filter_dict
        )

        # Форматирование результатов
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "prompt": doc.page_content,
                "category": doc.metadata.get("category", "General"),
                "for_devs": doc.metadata.get("for_devs", False),
                "similarity": 1.0 - score / 100.0  # Преобразуем дистанцию в сходство
            })

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Возвращает информацию о коллекции промптов

        Returns:
            Dict: Статистика коллекции
        """
        try:
            # Получение всех документов из коллекции
            docs = self.vectorstore.get()

            # Подсчет уникальных категорий
            categories = set()
            dev_count = 0

            if docs and hasattr(docs, "metadata"):
                for metadata in docs["metadatas"]:
                    if metadata and "category" in metadata:
                        categories.add(metadata["category"])
                    if metadata and metadata.get("for_devs", False):
                        dev_count += 1

            return {
                "count": len(docs["ids"]) if docs and hasattr(docs, "ids") else 0,
                "categories": list(categories),
                "categories_count": len(categories),
                "dev_prompts_count": dev_count
            }
        except Exception as e:
            print(f"Ошибка при получении статистики: {e}")
            return {
                "error": str(e)
            }
from transformers import AutoTokenizer
import re
from typing import List, Dict, Any
from prompt_database import PromptDatabase


class PromptGenerator:
    def __init__(self, db=None):
        """
        Инициализирует генератор промптов

        Args:
            db: Экземпляр базы данных промптов (опционально)
        """
        # Инициализация токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Инициализация базы данных промптов
        self.db = db if db else PromptDatabase()

        # Список стоп-слов для обработки запросов
        self.stopwords = [
            "the", "a", "an", "in", "on", "at", "for", "with", "by", "to", "and",
            "or", "of", "is", "are", "am", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "can", "could", "may", "might", "must", "about", "as",
            "from", "like", "that", "this", "there", "these", "those"
        ]

    def tokenize_and_process_query(self, query: str) -> str:
        """
        Токенизирует и обрабатывает запрос пользователя

        Args:
            query: Запрос пользователя

        Returns:
            str: Обработанный запрос для поиска
        """
        # Токенизация запроса
        tokens = self.tokenizer.tokenize(query)

        # Очистка токенов от спецсимволов и стоп-слов
        keywords = []
        for token in tokens:
            token = re.sub(r'[^\w\s]', '', token.lower())
            if token and token not in self.stopwords:
                keywords.append(token)

        # Если осталось слишком мало ключевых слов, используем исходный запрос
        if len(keywords) < 2:
            return query

        # Формирование запроса из ключевых слов
        enhanced_query = " ".join(keywords)
        return enhanced_query

    def detect_dev_related(self, query: str) -> bool:
        """
        Определяет, связан ли запрос с программированием или разработкой

        Args:
            query: Запрос пользователя

        Returns:
            bool: True, если запрос связан с разработкой
        """
        dev_keywords = [
            "code", "programming", "developer", "software", "app", "application",
            "framework", "library", "api", "server", "client", "database", "sql",
            "nosql", "frontend", "backend", "fullstack", "web", "mobile", "algorithm",
            "function", "class", "object", "variable", "python", "javascript", "java",
            "c++", "c#", "ruby", "php", "html", "css", "react", "angular", "vue",
            "node", "express", "django", "flask", "spring", "bootstrap", "jquery",
            "typescript", "git", "github", "gitlab", "aws", "azure", "cloud", "docker",
            "kubernetes", "devops", "selenium", "testing", "debug", "compile", "build",
            "deploy", "development", "programmer"
        ]

        lower_query = query.lower()

        # Проверяем наличие ключевых слов для разработчиков в запросе
        for keyword in dev_keywords:
            if keyword in lower_query:
                return True

        return False

    def generate_prompt_for_query(self, query: str) -> Dict[str, Any]:
        """
        Генерирует подходящий промпт на основе запроса пользователя

        Args:
            query: Запрос пользователя

        Returns:
            Dict: Сгенерированный промпт и метаданные
        """
        # Проверка минимальной длины запроса
        if len(query.strip()) < 3:
            return {
                "prompt": "Пожалуйста, уточните ваш запрос, чтобы я мог предложить подходящий промпт.",
                "category": "General",
                "for_devs": False,
                "similarity": 0.0
            }

        # Определяем, связан ли запрос с разработкой
        is_dev_related = self.detect_dev_related(query)

        # Обрабатываем запрос
        processed_query = self.tokenize_and_process_query(query)

        # Ищем подходящие промпты
        filter_dict = {"for_devs": is_dev_related} if is_dev_related else None
        relevant_prompts = self.db.search_prompts(
            processed_query,
            filter_dict=filter_dict,
            n_results=3
        )

        # Если нашли подходящие промпты, возвращаем лучший
        if relevant_prompts:
            best_prompt = relevant_prompts[0]
            return best_prompt

        # Если не нашли подходящих промптов, генерируем базовый промпт
        if is_dev_related:
            default_prompt = f"Я хочу, чтобы ты выступил в роли эксперта по {query}. Предоставь подробное объяснение, примеры кода и лучшие практики по этой теме."
        else:
            default_prompt = f"Я хочу, чтобы ты выступил в роли эксперта по теме {query}. Предоставь подробную информацию, включая ключевые аспекты, советы и рекомендации по этой теме."

        return {
            "prompt": default_prompt,
            "category": "Generated",
            "for_devs": is_dev_related,
            "similarity": 0.0
        }
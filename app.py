import pandas as pd
import argparse
from prompt_database import PromptDatabase
from prompt_generator import PromptGenerator


def load_prompts_from_csv(file_path):
    """Загружает промпты из CSV файла"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {e}")
        return None


def setup_database(csv_path=None):
    """Настройка базы данных промптов"""
    db = PromptDatabase()

    if csv_path:
        df = load_prompts_from_csv(csv_path)
        if df is not None:
            db.load_prompts_from_dataframe(df)

    # Отображаем статистику базы
    stats = db.get_collection_stats()
    print("\nСтатистика базы данных:")
    for key, value in stats.items():
        if key == "categories":
            print(f"Категории: {', '.join(value[:5])}... (всего {len(value)})")
        else:
            print(f"{key}: {value}")

    return db

def interactive_mode(generator):
    """Интерактивный режим работы с генератором промптов"""
    print("\n=== Интерактивный режим генератора промптов ===")
    print("Введите запрос, и я предложу подходящий промпт.")
    print("Введите 'exit' или 'quit' для выхода.")

    while True:
        user_query = input("\nВаш запрос: ")

        if user_query.lower() in ['exit', 'quit']:
            print("Завершение работы...")
            break

        if not user_query.strip():
            continue

        # Генерация промпта
        result = generator.generate_prompt_for_query(user_query)

        # Вывод результата
        print("\n" + "="*50)
        print(f"Категория: {result['category']}")
        print(f"Для разработчиков: {'Да' if result['for_devs'] else 'Нет'}")
        if result['similarity'] > 0:
            print(f"Релевантность: {result['similarity']:.2f}")
        print("="*50)
        print(result['prompt'])
        print("="*50)


# Добавляем отладочные сообщения
print("Приложение запущено")


def main():
    parser = argparse.ArgumentParser(description='Prompt generator tool')
    parser.add_argument('--csv', type=str, help='Path to CSV file with prompts')
    parser.add_argument('--query', type=str, help='Generate a prompt for this query')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()

    print(f"Аргументы командной строки: {args}")

    # Инициализируем базу данных промптов
    db = setup_database(args.csv)

    # Инициализируем генератор промптов
    generator = PromptGenerator(db)
    print("Генератор промптов инициализирован")

    # Если указан запрос в командной строке, генерируем промпт для него
    if args.query:
        print(f"\nГенерация промпта для запроса: '{args.query}'")
        result = generator.generate_prompt_for_query(args.query)

        print("\n" + "=" * 50)
        print(f"Категория: {result['category']}")
        print(f"Для разработчиков: {'Да' if result['for_devs'] else 'Нет'}")
        if result.get('similarity', 0) > 0:
            print(f"Релевантность: {result['similarity']:.2f}")
        print("=" * 50)
        print(result['prompt'])
        print("=" * 50)

    # Если указан флаг --interactive или не указаны другие параметры действия,
    # запускаем интерактивный режим
    elif args.interactive or not args.query:
        interactive_mode(generator)


if __name__ == "__main__":
    main()
    print("Приложение завершено")
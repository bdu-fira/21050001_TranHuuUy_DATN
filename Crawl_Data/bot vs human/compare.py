import re
import os

def parse_sentiment_file(filepath):
    """Parses a sentiment file into a dictionary of lists.

    Args:
        filepath (str): The path to the input text file.

    Returns:
        dict: A dictionary where keys are sentiments ('tích cực', 'tiêu cực',
              'trung tính') and values are lists of sentences.
              Returns None if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    data = {"tích cực": [], "tiêu cực": [], "trung tính": []}
    current_category = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Check for category lines (allow for optional space before colon)
                if line.lower().endswith(':'):
                    category_key = line[:-1].strip().lower()
                    if category_key in data:
                        current_category = category_key
                    else:
                        # If the line is like "1. tích cực:", extract the actual category
                        parts = category_key.split('.')
                        if len(parts) > 1:
                           potential_key = parts[-1].strip()
                           if potential_key in data:
                               current_category = potential_key
                           else:
                               # print(f"Warning: Unknown category '{line[:-1].strip()}' in {filepath}. Skipping.")
                               current_category = None # Reset if category is unknown
                        else:
                            # print(f"Warning: Unknown category '{line[:-1].strip()}' in {filepath}. Skipping.")
                            current_category = None # Reset if category is unknown

                # If we are inside a known category, look for quoted sentences
                elif current_category:
                    # Find all quoted strings on the line
                    sentences = re.findall(r'"(.*?)"', line)
                    for sentence in sentences:
                        cleaned_sentence = sentence.strip()
                        if cleaned_sentence: # Ensure we don't add empty strings
                            data[current_category].append(cleaned_sentence)

    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

    return data

# --- Main script ---
if __name__ == "__main__":
    file_bot = "bot sắp xếp.txt"
    file_gold = "người sắp xếp.txt"

    print(f"Comparing '{file_bot}' with '{file_gold}'...")

    bot_data = parse_sentiment_file(file_bot)
    gold_data = parse_sentiment_file(file_gold)

    if bot_data is None or gold_data is None:
        print("Cannot proceed due to file reading errors.")
    else:
        categories = ["tích cực", "tiêu cực", "trung tính"]
        results = []

        for category in categories:
            bot_sentences = bot_data.get(category, [])
            gold_sentences = gold_data.get(category, [])

            count_bot = len(bot_sentences)
            count_gold = len(gold_sentences)

            # Use sets for efficient comparison and handling duplicates within a list
            set_bot = set(bot_sentences)
            set_gold = set(gold_sentences)

            matching_sentences = set_bot.intersection(set_gold)
            count_match = len(matching_sentences)

            # Calculate percentage based on the bot's count, as in the image
            percentage = (count_match / count_bot * 100) if count_bot > 0 else 0.0

            # Store category name capitalized for display
            display_category = category.capitalize()
            results.append((display_category, count_gold, count_bot, count_match, percentage))

        # --- Print the results table ---
        print("\n" + "-" * 110)
        print(f"{'Loại câu':<15} {'Số câu trong dữ liệu vàng':<30} {'Số câu trong bot sắp xếp':<30} {'Số câu trùng nhau':<20} {'Tỉ lệ trùng':<15}")
        print("-" * 110)

        for category, count_g, count_b, count_m, perc in results:
            print(f"{category:<15} {count_g:<30} {count_b:<30} {count_m:<20} {perc:.1f}%")

        print("-" * 110)

        # Optional: Print sentences that didn't match for debugging
        print("\n--- Mismatched Sentences ---")
        for category in categories:
            bot_s = set(bot_data.get(category, []))
            gold_s = set(gold_data.get(category, []))
            print(f"\n>> {category.capitalize()} - In Bot but not Gold:")
            for s in sorted(list(bot_s - gold_s)):
                print(f'   - "{s}"')
            print(f"\n>> {category.capitalize()} - In Gold but not Bot:")
            for s in sorted(list(gold_s - bot_s)):
                print(f'   - "{s}"')
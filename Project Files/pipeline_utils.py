import re
import torch

dynamic_entities = {
    "SIZE": set(),
    "STYLE": set(),
    "TOPPING": set(),
    "QUANTITY": set(),
    "NUMBER": set(),
    "NOT_TOPPING": set(),
    "NOT_STYLE": set(),
    "VOLUME": set(),
    "DRINKTYPE": set(),
    "CONTAINERTYPE": set()
}

order_entities = ["PIZZAORDER", "DRINKORDER"]

entity_patterns = {
    "SIZE": r"\(SIZE ([^)]+)\)",
    "STYLE": r"\(STYLE ([^)]+)\)",
    "TOPPING": r"\(TOPPING ([^)]+)\)",
    "QUANTITY": r"\(QUANTITY ([^)]+)\)",
    "NUMBER": r"\(NUMBER ([^)]+)\)",
    "NOT_TOPPING": r"\(NOT \(TOPPING ([^)]+)\)",
    "NOT_STYLE": r"\(NOT \(STYLE ([^)]+)\)",
    "VOLUME": r"\(VOLUME ([^)]+)\)",
    "DRINKTYPE": r"\(DRINKTYPE ([^)]+)\)",
    "CONTAINERTYPE": r"\(CONTAINERTYPE ([^)]+)\)"
}

tag2id = {
    "O": 0,
    "B-NUMBER": 1,
    "I-NUMBER": 2,
    "B-SIZE": 3,
    "I-SIZE": 4,
    "B-STYLE": 5,
    "I-STYLE": 6,
    "B-TOPPING": 7,
    "I-TOPPING": 8,
    "B-QUANTITY": 9,
    "I-QUANTITY": 10,
    "B-NOT_TOPPING": 11,
    "I-NOT_TOPPING": 12,
    "B-NOT_STYLE": 13,
    "I-NOT_STYLE": 14,
    "B-VOLUME": 15,
    "I-VOLUME": 16,
    "B-DRINKTYPE": 17,
    "I-DRINKTYPE": 18,
    "B-CONTAINERTYPE": 19,
    "I-CONTAINERTYPE": 20,
}

# id2tag = {v:k for k, v in tag2id.items()}
# Create reverse mapping from ID to tag
id2tag = {v: k for k, v in tag2id.items()}

IS_tag2id = {
    "O": 0,
    "B-PIZZAORDER": 1,
    "I-PIZZAORDER": 2,
    "B-DRINKORDER": 3,
    "I-DRINKORDER": 4
}

IS_id2tag = {v: k for k, v in IS_tag2id.items()}

pizza_pattern = r'\((?=PIZZAORDER)'
drink_pattern = r'\((?=DRINKORDER)'


def find_order_indices(text):
    # Find start positions of PIZZAORDER and DRINKORDER
    order_positions = []

    # Find all PIZZAORDER positions
    for match in re.finditer(pizza_pattern, text):
        order_positions.append((match.start(), "PIZZAORDER"))

    # Find all DRINKORDER positions
    for match in re.finditer(drink_pattern, text):
        order_positions.append((match.start(), "DRINKORDER"))

    # Sort by position
    order_positions.sort()

    return order_positions


def extract_order_spans(text):
    spans = []
    order_positions = find_order_indices(text)
    for start_pos, order_type in order_positions:
        stack = []
        nesting_level = 0

        # Scan forward from start position
        for i in range(start_pos, len(text)):
            if text[i] == '(':
                nesting_level += 1
                stack.append(i)
            elif text[i] == ')':
                nesting_level -= 1
                stack.pop()

                # Found matching closing parenthesis
                if nesting_level == 0:
                    # Extract text between parentheses
                    span_text = text[start_pos:i + 1]
                    spans.append((start_pos, i + 1, span_text, order_type))
                    break

    return spans

def tag_orders(text):
    # Get order spans
    spans = extract_order_spans(text)

    # Split into words and track positions
    words = text.split()
    word_positions = []
    current_pos = 0

    for word in words:
        word_positions.append((current_pos, current_pos + len(word)))
        current_pos = text.find(word, current_pos) + len(word)

    # Initialize tags
    tags = ['O'] * len(words)

    # Tag words based on spans
    for span_start, span_end, span_text, order_type in spans:
        first_word = True
        for i, (word_start, word_end) in enumerate(word_positions):
            if span_start <= word_start and word_end <= span_end:
                if first_word and not words[i].startswith(('(', ')')):
                    tags[i] = f'B-{order_type}'
                    first_word = False
                elif not words[i].startswith(('(', ')')):
                    tags[i] = f'I-{order_type}'

    # Filter out brackets and special tokens
    clean_tokens = []
    clean_tags = []
    for word, tag in zip(words, tags):
        if not (word.startswith('(') or word.startswith(')') or word in ['ORDER', 'PIZZAORDER', 'DRINKORDER']):
            clean_tokens.append(word)
            clean_tags.append(tag)
    tag_ids = [IS_tag2id[tag] for tag in clean_tags]
    return clean_tokens, tag_ids, clean_tags
def group_order_tokens(tokens, tags):
    grouped_tokens = []
    current_group = []

    for token, tag in zip(tokens, tags):
        # Start of new order
        if tag.startswith('B-'):
            if current_group:
                grouped_tokens.append(current_group)
            current_group = [token]
        # Inside current order or O tag
        elif tag.startswith('I-'):
            current_group.append(token)
        else:  # O tag
            if current_group:
                grouped_tokens.append(current_group)
            current_group = []

    # Add last group if exists
    if current_group:
        grouped_tokens.append(current_group)

    return grouped_tokens

def parse_top_string(text, patterns, tag2id):
    # Split text into words
    words = text.split()

    # Initialize lists for final output
    clean_tokens = []
    final_tags = []

    # Initialize all words as "O" tag
    tags = ["O"] * len(words)

    # Find all entities
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)

        for match in matches:
            entity_value = match.group(1)
            entity_words = entity_value.split()

            # Find position of these words in original text
            for i, word in enumerate(words):
                if word == entity_words[0]:
                    # Check if this is the start of our entity
                    if all(words[i + j] == entity_words[j] for j in range(len(entity_words))):
                        # First word gets B- tag
                        tags[i] = f"B-{entity_type}"
                        # Subsequent words get I- tag
                        for j in range(1, len(entity_words)):
                            tags[i + j] = f"I-{entity_type}"

    # Filter out special characters and structural keywords
    for word, tag in zip(words, tags):
        if not (word.startswith('(') or word.startswith(')')):
            clean_tokens.append(word)
            final_tags.append(tag2id[tag])
    return clean_tokens, final_tags
def group_corresponding_tags(grouped_tokens, all_tokens, all_tags_or_ids):
    grouped_values = []

    for token_group in grouped_tokens:
        current_group = []
        # Find start index of current group in all_tokens
        start_search_idx = 0

        for token in token_group:
            # Find token in remaining portion of all_tokens
            for i in range(start_search_idx, len(all_tokens)):
                if all_tokens[i] == token:
                    current_group.append(all_tags_or_ids[i])
                    start_search_idx = i + 1
                    break

        grouped_values.append(current_group)

    return grouped_values


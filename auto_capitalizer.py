"""
Auto-Capitalization and Natural Language Formatting Module
Applies automatic capitalization and grammatical rules:
1. Capitalizes start of sentences (. ! ?)
2. Capitalizes standalone 'i' -> 'I'
3. Auto-formats leading character of text or sentence
"""

import re


class AutoCapitalizer:
    @staticmethod
    def process_notepad_text(raw_text):
        """
        Processes raw text stream from air typing/notepad and applies NLP rules:
        1. Capitalizes first non-space character of text.
        2. Capitalizes first character following sentence enders (. ! ?).
        3. Capitalizes standalone 'i' -> 'I'.
        """
        if not raw_text or len(raw_text) == 0:
            return raw_text

        # 1. Capitalize first non-space character of text
        first_char_index = 0
        while first_char_index < len(raw_text) and raw_text[first_char_index] == ' ':
            first_char_index += 1

        if first_char_index < len(raw_text):
            raw_text = (
                raw_text[:first_char_index] 
                + raw_text[first_char_index].upper() 
                + raw_text[first_char_index + 1:]
            )

        # 2. Capitalize first character following sentence enders (. ! ?)
        raw_text = re.sub(
            r'([.!?]\s*)([a-z])', 
            lambda m: m.group(1) + m.group(2).upper(), 
            raw_text
        )

        # 3. Capitalize standalone 'i' -> 'I'
        raw_text = re.sub(r'\bi\b', 'I', raw_text)

        return raw_text

    @staticmethod
    def format_text(text):
        """Alias for process_notepad_text to maintain full backward compatibility."""
        return AutoCapitalizer.process_notepad_text(text)

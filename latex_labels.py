# latex_labels.py

import re
import random
import string
import asyncio

from config import Config

class LabelManager:
    def __init__(self, document):
        """
        Initialize the latex_labels class with a LaTeX document string.
        It collects all existing labels from occurrences of \\label{X}
        and stores them in a set to avoid duplication.
        """
        pattern = r'\\label\{([^}]+)\}'
        self.existing_labels = set(re.findall(pattern, document))
        self.lock = asyncio.Lock()

    async def get_label(self):
        """
        Asynchronously generates and returns a new unique label.
        The label is a 4-character string where each character is
        a digit (0-9) or a letter (a-z, A-Z).
        """
        characters = string.ascii_letters + string.digits
        while True:
            new_label = ''.join(random.choices(characters, k=Config.NUM_LABEL_CHAR))
            async with self.lock:
                if new_label not in self.existing_labels:
                    self.existing_labels.add(new_label)
                    return new_label

    async def check_label(self, label: str) -> bool:
        """
        Asynchronously checks if the provided label exists.
        
        Parameters:
            label (str): The label to check.
        
        Returns:
            bool: True if the label exists, False otherwise.
        """
        async with self.lock:
            return label in self.existing_labels

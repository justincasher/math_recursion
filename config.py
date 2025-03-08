# config.py

import os

class Config:
    """
    Stores fixed configuration values for the LLM application.
    These values can be edited directly in this file when needed.
    """

    # The default model name to use for Gemini requests
    DEFAULT_MODEL_NAME = "gemini-2.0-flash"

    # Your Gemini API key
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

    # Server backoff for overload
    MAX_RETRIES = 5        # Number of total attempts
    BACKOFF_FACTOR = 1.0   # Base wait time; can be adjusted as needed

    # Document to edit
    DOC_LOAD = "test_problems/Putnam_2024_A6.txt"
    DOC_SAVE = "math_new.txt"   

    # Whether or not to use parallel calls (debugging)
    PARALLEL = True

    # Number of steps at each
    L1_REASONING_STEPS = 5
    L2_REASONING_STEPS = 2
    L3_REASONING_STEPS = 2
    L4_REASONING_STEPS = 2

    # This creates multiple bots at each level, so it is not recommended 
    # to choose many bots for each. For reviewers, it means that we 
    # have that many check the math, and any rejection means we try again. 
    # For every other bot type, it means we try that many times at creating 
    # the desired section, subsection, math, etc., and then perform a round 
    # robin tournament to determine the best result.
    NUM_L2_BOTS = 1
    NUM_L3_BOTS = 1
    NUM_L4_BOTS = 3
    NUM_REVIEWERS = 3

    # Control flags for console output at different debug levels
    L1_PRINT = False
    L2_PRINT = False
    L3_PRINT = False
    L4_PRINT = False
    L4_REVIEW_PRINT = False

    # Whether to open the visualizer
    VISUALIZER = True 
    MAX_DISPLAY = 800 # Pixels
    INSTRUCTION_HEIGHT = 250 # Pixels

    # Label generation
    NUM_LABEL_CHAR = 4

    @classmethod
    def as_string(cls) -> str:
        """
        Returns all configuration variables in the Config class as a string.
        """
        config_str = "\n----------- CONFIG -----------\n"
        for attribute_name in dir(cls):
            # Skip internal (dunder) attributes and builtins
            if not attribute_name.startswith("__"):
                attribute_value = getattr(cls, attribute_name)
                # Avoid callable attributes (like methods)
                if not callable(attribute_value):
                    config_str += f"{attribute_name} = {attribute_value}\n"
        config_str += "-" * 30 + "\n"
        return config_str
# clean_llm_output.py

def clean_llm_output(llm_output: str) -> str:
    """
    Cleans the output string from an LLM by removing unwanted sections.
    
    This function performs the following operations:
      - Removes lines starting with "You are on Step" and stops processing further lines.
      - Removes any trailing blank lines.
      - Strips triple backticks from the first and last lines if they are present.
      - Removes the final line if it starts with "final answer:" (case-insensitive).
    
    Args:
        llm_output (str): The original LLM output string.
    
    Returns:
        str: The cleaned output string.
    """
    # Split the output into lines.
    lines = llm_output.splitlines()
    
    # Remove lines after encountering a line that starts with "You are on Step"
    cleaned_lines = []
    bad_phrases = {
        "You are on Step",
        "TASK:",
        "Final Answer:",
        "MATH BLOCKS",
        "SUBSECTION INSTRUCTIONS",
        "NEXT STEP:"
    }
    for line in lines:
        if any(line.lower().startswith(bp.lower()) for bp in bad_phrases):
            break
        cleaned_lines.append(line)
    
    # Remove trailing blank lines.
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    # Remove triple backticks from the first and last lines if present.
    if cleaned_lines and "```" in cleaned_lines[0]:
        cleaned_lines = cleaned_lines[1:]
    if cleaned_lines and "```" in cleaned_lines[-1]:
        cleaned_lines = cleaned_lines[:-1]
    
    # Join the cleaned lines into a single string.
    return "\n".join(cleaned_lines)
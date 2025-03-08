# bots/review_bot.py

from typing import Optional

from llm_call import llm_call
from config import Config


class ReviewBot:
    def __init__(
        self, 
        document: str, 
        section: str, 
        subsection: str, 
        instruction: str, 
        environment_block: str
    ):
        """
        Initialize the ReviewBot instance for multi-step mathematical reasoning review.
        """
        # Document and instruction details
        self.document = document
        self.section = section
        self.subsection = subsection
        self.instruction = instruction
        self.environment_block = environment_block

        # Unified system prompt for all review steps
        self.system_prompt = (
            "You are the Reasoning Reviewer Bot. You are writing a LaTeX document with other bots, and you are "
            "responsible for performing a multi-step review of their mathematical reasoning.\n\n"

            "You will go through a multi-step process composed of the following:\n"
            "1. Sentence-by-sentence analysis.\n"
            "2. Claim verification.\n"
            "3. Creating a numbered list of potential errors.\n"
            "4. Verifying (confirm or dismiss) each potential error.\n"
            "5. Clarification of confirmed errors.\n"
            "6. Final summary paragraph indicating whether the math is acceptable or needs revision, "
            "ending with either 'ACCEPT' or 'REJECT'.\n\n"

            "Notes:\n"
            "- Do NOT attempt or respond about any other steps than the one you are on.\n"
            "- Never use numerical tools such as code (Python), WolframAlpha, OEIS, etc."
        )

        # State variables for review steps
        self.sentence_logic_analysis: Optional[str] = None
        self.claims_verification: Optional[str] = None
        self.error_list: Optional[str] = None
        self.verified_errors: Optional[str] = None
        self.clarifications: Optional[str] = None
        self.summary: str = "N/A"

        self.done = False
        self.accepted = False
        self.iterations = 0
        self.current_llm_call_index = 0


    async def _sentence_logic_analysis(self):
        """
        Step 1A: Analyze the logic of each sentence.
        For each sentence, re-check the logic, calculations, and notation. 
        End each sentence's analysis with "CORRECT" or "FALSE" and note any logical mismatches with the instructions.
        """
        prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "CURRENT SUBSECTION:\n\n"

            f"{self.subsection}\n\n"

            "MATH INSTRUCTION:\n\n"

            f"{self.instruction}\n\n"

            "MATH TO CHECK:\n\n"

            f"{self.environment_block}\n\n"

            "You are on Step 1: Analyze each sentence for logical consistency.\n\n"

            "For each sentence in the math to check, do the following:\n "
            "A) Write out the sentence.\n"
            "B) Reason about whether it is correct.\n"
            "C) At the end of each sentence's analysis, write 'CORRECT' or 'FALSE'."
        )

        self.sentence_logic_analysis = await llm_call(
            prompt=prompt, 
            system_prompt=self.system_prompt
        )

        if Config.L4_REVIEW_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"Review Bot Iteration #{self.iterations + 1} - Step 1 - Sentence Logic Analysis\n" +
                "-" * 50 + "\n" +
                f"{self.sentence_logic_analysis}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _claims_verification(self):
        """
        Step 2: Verify that all claims are proven.
        For every claim made in the document, check that it has an accompanying proof (unless it is a well-known result or a conjecture).
        Report any unproven claims or mismatches.
        """
        prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "CURRENT SUBSECTION:\n\n"

            f"{self.subsection}\n\n"

            "MATH INSTRUCTION:\n\n"

            f"{self.instruction}\n\n"

            "MATH TO CHECK:\n\n"

            f"{self.environment_block}\n\n"

            "You are on Step 2: Verify that all claims made in the math to check have indeed been proven.\n\n"

            "For each claim made in any theorem, proposition, lemma, etc. do the following:\n" 
            "A) Write out the claim."
            "B) Reason about whether or not claim has a complete proof."
            "C) Write a 'CORRECT' to indicate if the claim is proven or does not need proof, else 'False'c.\n\n"
            
            "Finally, note any mismatches with the provided instruction and the math to check. Be lenient."
        )

        self.claims_verification = await llm_call(
            prompt=prompt, 
            system_prompt=self.system_prompt
        )

        if Config.L4_REVIEW_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"Review Bot Iteration #{self.iterations + 1} - Step 2 - Claims Verification\n" +
                "-" * 50 + "\n" +
                f"{self.claims_verification}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _list_errors(self):
        """
        Step 3: Create a numbered list of potential errors.
        Based on the analyses from Steps 1A and 1B, list potential errors.
        If no errors are identified, explicitly state so.
        """
        prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "CURRENT SUBSECTION:\n\n"

            f"{self.subsection}\n\n"

            "MATH INSTRUCTION:\n\n"

            f"{self.instruction}\n\n"

            "MATH TO CHECK:\n\n"

            f"{self.environment_block}\n\n"

            "SENTENCE LOGIC ANALYSIS:\n\n"

            f"{self.sentence_logic_analysis}\n\n"

            "CLAIMS VERIFICATION:\n\n"
            
            f"{self.claims_verification}\n\n"

            "You are on Step 3: Create a numbered list of potential errors based on the above analyses.\n\n"

            "Based on the sentence logic analysis and claim verification, create a list of all potential errors " 
            "in the math to check. If no errors are identified, state explicitly that there are no errors."
        )

        self.error_list = await llm_call(
            prompt=prompt, 
            system_prompt=self.system_prompt
        )

        if Config.L4_REVIEW_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"Review Bot Iteration #{self.iterations + 1} - Step 3 - List Potential Errors\n" +
                "-" * 50 + "\n" +
                f"{self.error_list}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _verify_errors(self):
        """
        Step 4: Verify each potential error.
        For each error in the list, decide whether it is CONFIRMED (a genuine error) or DISMISSED, 
        providing a short collaborative explanation.
        """
        prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "CURRENT SUBSECTION:\n\n"

            f"{self.subsection}\n\n"

            "MATH INSTRUCTION:\n\n"

            f"{self.instruction}\n\n"

            "MATH TO CHECK:\n\n"

            f"{self.environment_block}\n\n"

            "POTENTIAL ERRORS:\n\n"

            f"{self.error_list}\n\n"

            "You are on Step 4: Verify (confirm or dismiss) each potential error.\n\n"

            "Another bot has identified the above as being potential errors. However, you need "
            "to be critical and truly determine whether these are indeed errors or not.\n\n"

            "For each potential error, do the following:\n"
            "A) Reason about the validity of the objection.\n"
            "B) Write 'CONFIRMED' or 'DISMISSED'.\n\n"

            "If there are no errors, simply write 'NO ERRORS'."
        )

        self.verified_errors = await llm_call(
            prompt=prompt, 
            system_prompt=self.system_prompt
        )

        if Config.L4_REVIEW_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"Review Bot Iteration #{self.iterations + 1} - Step 4 - Verify Errors\n" +
                "-" * 50 + "\n" +
                f"{self.verified_errors}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _final_summary(self):
        """
        Step 6: Produce a final summary.
        Write a summary that references any confirmed errors and highlights correct portions, ending with either 'ACCEPT' or 'REJECT'.
        """
        prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "CURRENT SUBSECTION:\n\n"

            f"{self.subsection}\n\n"

            "MATH INSTRUCTION:\n\n"

            f"{self.instruction}\n\n"

            "MATH TO CHECK:\n\n"

            f"{self.environment_block}\n\n"

            "ERROR LIST:\n\n"

            f"{self.error_list}\n\n"

            "ERROR VERIFICATION RESULTS:\n\n"

            f"{self.verified_errors}\n\n"

            "You are on Step 6: Write a final summary of the review.\n\n"

            "Reference any confirmed errors from the error list. "
            "Provide direct quotes of the problematic text and explain the error you found. "
            "Use collaborative language in your summary (e.g., 'We believe there may be an issue with...')."
            "Do NOT make any affirmative statements about what correct values, claims, or proofs would be.\n\n"

            "Finally, end with a single line containing either 'ACCEPT' or 'REJECT'."
        )

        self.summary = await llm_call(
            prompt=prompt,
            system_prompt=self.system_prompt
        )

        # Set accepted flag based on the final line of the summary.
        final_line = self.summary.strip().splitlines()[-1].strip()
        if "ACCEPT" in final_line.upper():
            self.accepted = True

        if Config.L4_REVIEW_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"Review Bot Iteration #{self.iterations + 1} - Step 6 - Final Summary\n" +
                "-" * 50 + "\n" +
                f"{self.summary}\n" +
                "=" * 50 + "\n"
            )
            print(output)
            

    async def step(self):
        """
        Execute the next review step.
        The review process now includes two separate steps for sentence analysis.
        """
        if self.done:
            raise RuntimeError("ReviewBot has already completed all review steps.")

        llm_call_sequence = [
            self._sentence_logic_analysis,
            self._claims_verification,
            self._list_errors,
            self._verify_errors,
            self._final_summary,
        ]

        await llm_call_sequence[self.current_llm_call_index]()
        self.iterations += 1
        self.current_llm_call_index += 1

        if self.current_llm_call_index >= len(llm_call_sequence):
            self.done = True
            self.current_llm_call_index = 0

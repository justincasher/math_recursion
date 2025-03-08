# bots/L4_bot.py

import asyncio
import re
from typing import Optional

from bots.review_bot import ReviewBot
from config import Config
from latex_labels import LabelManager
from llm_call import llm_call


class L4Bot:
    def __init__(
            self, 
            document: str, 
            section: str, 
            subsection: str, 
            L2_instruction: str,
            L3_instruction: str,
            L4_instruction: str,
            lbl_mgr: LabelManager
        ):
        """
        Initialize the L4Bot instance.

        This bot is responsible for generating a specific LaTeX environment block based on provided 
        instructions. Operating within a collaborative multi-bot system for composing a mathematics paper, 
        the L4Bot iteratively produces chain-of-thought reasoning, generates valid LaTeX code for mathematical 
        constructs (e.g., definitions, theorems, proofs, lemmas, corollaries, examples), and initiates a parallel 
        review process via ReviewBot instances to ensure correctness and quality.
        """
        # Prompting and document information
        self.document = document
        self.section = section
        self.subsection = subsection
        self.L2_instruction = L2_instruction
        self.L3_instruction = L3_instruction
        self.L4_instruction = L4_instruction
        self.lbl_mgr = lbl_mgr

        self.system_prompt = (
            "You are a Level 4 (L4) bot. You are collaborating with other bots to write a mathematics paper, "
            "and you are responsible for writing the actual mathematics. This might be "
            "definitions, lemmas, theorems, proofs, corollaries, examples, etc. You must always include a proof "
            "for any theorem, proposition, or lemma. You will be given instructions describing what to write. "
            "You must produce valid LaTeX for the requested environment. If referencing other parts of the document, "
            "maintain consistency with any previously introduced labels or notation.\n\n"

            "You will iterate through a multi-step process composed of the following:\n"
            "1. Writing mathematics according to the instructions.\n"
            "2. Reviewing the mathematics that you wrote.\n"
            "3. Create a list of any mistakes in the mathematics.\n\n"

            "Notes:\n"
            "- Use LaTeX when writing math, but NEVER write out an entire document, just the relevant text.\n"
            r"- Use $...$ instead of \(...\)." "\n"
            "- Use LaTeX environments like gather, theorem, align, lemma, proof, example, etc.\n"
            "- Do NOT attempt or respond about any other steps than the one your are on.\n"
            "- Never use numerical tools (i.e., methods) such as code (Python), WolframAlpha, OEIS, etc."
        )

        # Initialize state variables
        self.raw_reasoning = "N/A"
        self.math_draft = "N/A"
        self.review_summary = "N/A"

        # Parallel review bots (for later evaluation)
        self.children = []

        # Control attributes for iterative processing
        self.iterations = 0
        self.done = False
        self.incomplete = True
        self.current_llm_call_index = 0


    async def _chain_of_thought_reasoning(self):
        """
        LLM call to produce chain-of-thought reasoning.
        This method prompts the bot to continue thinking step-by-step (using its previous attempts and feedback)
        about how to produce the requested mathematics.
        """
        reasoning_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "SUBSECTION INSTRUCTIONS:\n\n"

            f"{self.L3_instruction}\n\n"

            "CURRENT SUBSECTION:\n\n"

            f"{self.subsection}\n\n"

            "MATH INSTRUCTIONS:\n\n"

            f"{self.L4_instruction}\n\n"

            "PREVIOUS REASONING:\n\n"

            f"{self.raw_reasoning}\n\n"

            "PREVIOUS ATTEMPT:\n\n"

            f"{self.math_draft}\n\n"

            "REVIEWER FEEDBACK:\n\n"

            f"{self.review_summary}\n\n"

            "TASK:\n\n"

            "You are on Step 1: Writing mathematics according to the instructions.\n\n"
            
            "This is your scratch paper where you can reason about how to produce the "
            "instructed mathematics. Note that you are not writing out an entire subsection, but "
            "rather writing out a piece of mathematics that will be inserted into the current "
            "subsection. Be sure to be rigorous and think about how to show every step of "
            "mathematics when proving or asserting anything. Remember to "
            "never use numerical tools such as code (Python), WolframAlpha, OEIS, etc."
        )

        self.raw_reasoning = await llm_call(
            prompt=reasoning_prompt, 
            system_prompt=self.system_prompt
        )
        
        if Config.L4_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L4 Bot Iteration #{self.iterations + 1} - Chain-of-thought reasoning:\n" +
                "-" * 50 + "\n" +
                f"{self.raw_reasoning}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _generate_environment_block(self):
        """
        LLM call to generate the final LaTeX environment block.
        This method takes the chain-of-thought reasoning and produces one complete LaTeX code block
        for the requested mathematics.
        After the call, it also instantiates the parallel ReviewBot tasks.
        """
        latex_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "SUBSECTION INSTRUCTIONS:\n\n"

            f"{self.L3_instruction}\n\n"

            "CURRENT SUBSECTION:\n\n"

            f"{self.subsection}\n\n"

            "MATH INSTRUCTIONS:\n\n"

            f"{self.L4_instruction}\n\n"
            
            "REASONING:\n\n"

            f"{self.raw_reasoning}\n\n"

            "You are on Step 1: Writing mathematics according to the instructions.\n\n"

            "Your objective is to convert your scratch work/reasoning into a formal piece " 
            "of mathematics written in LaTeX. You should be cleaning up the work: While " 
            "you want to show every step, you should be putting it into a format that " 
            "could be added to a textbook.\n\n"

            r"Always use proper LaTeX environments by using \begin{X}...\end{X} " 
            "where X can be theorem, proposition, lemma, example, proof, remark, definition, " 
            "corollary, etc. whenever appropriate. Each mathematical statement should be "
            "properly formatted in its corresponding environment.\n\n"

            "Furthermore, whenever you write out a theorem, proposition, lemma, etc., you should "
            r"always include a proof using the \begin{proof}...\end{proof} environment. Never " 
            "skip any computations and show all of your work.\n\n" 
            
            "Note that you are not writing out an entire subsection, but rather writing out a " 
            "piece of mathematics that will be inserted into the current subsection. Hence, " 
            "Do not attempt to write out the entire document, section, or subsection. Just the "
            "instructed mathematics."
        )

        self.math_draft = await llm_call(
            prompt=latex_prompt, 
            system_prompt=self.system_prompt
        )

        if Config.L4_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L4 Bot Iteration #{self.iterations + 1} - Raw LaTeX:\n" +
                "-" * 50 + "\n" +
                f"{self.math_draft}\n" +
                "=" * 50 + "\n"
            )
            print(output)

        # Instantiate ReviewBot tasks for parallel evaluation.
        self.children = [
            ReviewBot(self.document, self.section, self.subsection, self.L4_instruction, self.math_draft)
            for _ in range(Config.NUM_REVIEWERS)
        ]


    async def _review_evaluation(self):
        """
        LLM call to evaluate the generated environment block based on reviewer feedback.
        This method gathers the summaries from the ReviewBot children and prompts an LLM to decide 
        whether the environment block is acceptable.
        """
        # Concatenate reviewer summaries. (Assume each review bot has produced a summary in its 'summary' attribute.)
        self.enumerated_feedback = "\n".join(
            f"Reviewer {i+1}: {reviewer.summary}" for i, reviewer in enumerate(self.children)
        )

        review_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "CURRENT SUBSECTION:\n\n"

            f"{self.subsection}\n\n"

            "GENERATED MATH BLOCK:\n\n"

            f"{self.math_draft}\n\n"

            "REVIEWER FEEDBACK:\n\n"

            f"{self.enumerated_feedback}\n\n"

            "TASK:\n\n"

            "You are on Step 3: Create a list of any mistakes in the mathematics.\n\n"

            "Create a non-repeating list of all of the errors that the reviewers found using "
            "1), 2), etc. to mark each possible mistake. Use collaborative language such as "
            "'we believe that...', 'we think that..,' etc. Provide direct citations of the math " 
            "that is believed to be incorrect."
        )

        self.review_summary = await llm_call(
            prompt=review_prompt, 
            system_prompt=self.system_prompt
        )

        if Config.L4_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L4 Bot Iteration #{self.iterations + 1} - Review summary:\n" +
                "-" * 50 + "\n" +
                f"{self.review_summary}\n" +
                "=" * 50 + "\n"
            )
            print(output)

        self.iterations += 1

        #  Updated check: if every child's accepted attribute is True, then mark as done.
        if all(reviewer.accepted for reviewer in self.children):
            self.done = True
            self.incomplete = False
        # If we ran out of iterations, mark as done but incomplete
        if not self.iterations < Config.L4_REASONING_STEPS:
            self.done = True
            self.incomplete = True


    async def step(self):
        """
        Execute the next LLM call in the sequence.
        Each call to this method triggers exactly one llm_call.
        """
        if self.done:
            raise RuntimeError("L4 bot called after being marked done.")
        if self.iterations >= Config.L4_REASONING_STEPS:
            raise RuntimeError("Iteration limit reached: Maximum number of L4 reasoning steps exceeded.")

        llm_call_sequence = [
            self._chain_of_thought_reasoning,
            self._generate_environment_block,
            self._review_evaluation,
        ]

        await llm_call_sequence[self.current_llm_call_index]()
        
        self.current_llm_call_index += 1

        if self.current_llm_call_index >= len(llm_call_sequence):
            self.current_llm_call_index = 0
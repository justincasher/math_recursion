# bots/L3_bot.py

import asyncio
from collections import defaultdict
import random
import re

from bots.L4_bot import L4Bot
from config import Config
from latex_labels import LabelManager
from llm_call import llm_call


class L3Bot:
    def __init__(
            self, 
            document: str, 
            section: str, 
            subsection_draft: str,
            L1_instruction: str,
            L2_instruction: str, 
            L3_instruction: str,
            lbl_mgr: LabelManager
        ):
        """
        Initialize the L3Bot instance.

        This bot is responsible for constructing a specific LaTeX subsection based on 
        instructions from an L2 bot. It determines necessary mathematical components 
        (e.g., theorems, lemmas, examples), delegates their creation to L4 bots, and 
        iteratively refines the subsection until it reaches a satisfactory state.
        """
        self.document = document
        self.section = section
        self.L1_instruction = L1_instruction
        self.L2_instruction = L2_instruction 
        self.L3_instruction = L3_instruction
        self.subsection_draft = subsection_draft
        self.lbl_mgr = lbl_mgr

        self.system_prompt = (
            "You are a Level 3 (L3) bot. You are writing a LaTeX document with other bots, and you are " 
            r"responsible for a single \subsection{...}. You received instructions on what this section should contain "  
            "from an L2 bot. You will delegate any computational work to an L4 bot. In particular, any time you "
            "would like to write and prove a statement (theorem, proposition, lemma), compute an example, write "
            "a conjecture, etc., you must delegate this to an L4 bot. "
            "These L4 bots will work in *parallel*, meaning you should not propose changes that depend on eachother.\n\n"

            "You will iterate through a multi-step process composed of the following:\n"
            r"1. Figuring out what needs be added next to the subsection." "\n"
            r"2. Writing instructions for the L4 bots on what math to construct in parallel." "\n"
            r"3. Drafting a copy of the \subsection{...} by inserting the work of the L4 bots." "\n"
            r"4. Deciding whether the \subsection{...} is in satisfactory shape or needs to go through further revisions." "\n\n"

            "Notes:\n"
            "- Be economical about what you write, always considering how it relates to the instructions.\n"
            "- Use LaTeX when writing math, but NEVER write out an entire document, just the relevant text.\n"
            r"- Use $...$ and $$...$$ instead of \(...\) and \[...\]."
            "- Use LaTeX environments like gather, theorem, align, lemma, proof, example, etc.\n"
            "- Never use numerical tools (i.e., methods) such as code (Python), WolframAlpha, OEIS, etc."
        )

        # Variables for pre-parallel processing
        self.step_a_output = ""
        self.step_b_output = ""
        self.step_c_output = ""
        self.prelim_reasoning_response = ""
        self.reasoning_response = ""
        self.formatted_instructions_response = ""
        self.environment_instructions = []
        self.children = []  # List of L4Bot instances
        self.round_robin = [] # List of L4RoundRobin instances
        self.restart = False

        # Variables for post-parallel processing
        self.math_drafts = {}  # Outputs from L4 bots
        self.final_decision_response = ""

        # Control attributes
        self.iterations = 0
        self.done = False
        self.current_llm_call_index = 0

    async def _reasoning_step_A(self):
        """Step A: Reason about what math could be added."""
        step_a_prompt = (
            "DOCUMENT INSTRUCTIONS:\n\n"

            f"{self.L1_instruction}\n\n"

            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS:\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "SUBSECTION INSTRUCTIONS:\n\n"

            f"{self.L3_instruction}\n\n"

            "CURRENT SUBSECTION (working draft):\n\n"

            f"{self.subsection_draft}\n\n"

            "TASK:\n\n"

            "You are on Step 1: Figuring out what needs be added next to the subsection.\n\n"  
            
            "Reason about what math could be added next to or improved upon within the current subsection. "
            "Use this space as a scratchpad for figuring out what can be added to the current subsection "
            "in parallel. This means that you should think about both what math should be added "
            "and whether it should be added one at a time, or if multiple parts can be added at once.\n\n"

            "Do NOT write any instructions here or work on any other steps."
        )
        
        self.step_a_output = await llm_call(
            prompt=step_a_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L3_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L3 Bot Iteration #{self.iterations + 1} - Step A (Reasoning what math to add):\n" +
                "-" * 50 + "\n" +
                f"{self.step_a_output}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _reasoning_step_B(self):
        """Step B: Propose new, logically independent math to be added."""
        step_c_prompt = (
            "DOCUMENT INSTRUCTIONS:\n\n"

            f"{self.L1_instruction}\n\n"

            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS:\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "SUBSECTION INSTRUCTIONS:\n\n"

            f"{self.L3_instruction}\n\n"

            "CURRENT SUBSECTION (working draft):\n\n"

            f"{self.subsection_draft}\n\n"

            "PREVIOUS OUTPUT (Step A):\n\n"

            f"{self.step_a_output}\n\n"

            "TASK:\n\n"

            "You are on Step 1: Figuring out what needs be added next to the subsection.\n\n"  
            
            "You just reasoned about what math could be added next to the current subsection. "
            "Remember: The math will be carried out in parallel, so it is important not to propose "
            "something that would benefit or need the completion of another piece of mathematics.\n\n"

            "Based on your previous reasoning, propose new math that is logically independent and should be written next. "
            "Clearly specify the type (e.g., theorem, lemma, example, etc.) and a description of what it should contain, as well as what "
            "should NOT be included (i.e., what is being delegated to the other bots).\n\n"

            "After you have proposed the new math, go through item by item and make sure they are indeed "
            "independent instructions for things to add.\n\n"

            "Examples of instruction X depending on instruction Y are:\n"
            "- If instruction X uses a definition from instruction Y.\n"
            "- If instruction X builds on examples introduced in instruction Y.\n"
            "- If instruction X references results from instruction Y.\n"
            "- If instruction X proves a conjecture from instruction Y."
            "- etc.\n\n"

            "Do NOT write any instructions here or work on any other steps."
        )

        self.step_b_output = await llm_call(
            prompt=step_c_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L3_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L3 Bot Iteration #{self.iterations + 1} - Step B (Proposing new math):\n" +
                "-" * 50 + "\n" +
                f"{self.step_b_output}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _reasoning_step_C(self):
        """Step C: Double-check that the proposed math is coherent and revise if needed."""
        step_c_prompt = (
            "DOCUMENT INSTRUCTIONS:\n\n"

            f"{self.L1_instruction}\n\n"

            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS:\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "SUBSECTION INSTRUCTIONS:\n\n"

            f"{self.L3_instruction}\n\n"

            "CURRENT SUBSECTION (working draft):\n\n"

            f"{self.subsection_draft}\n\n"

            "PREVIOUS OUTPUT (Step A):\n\n"

            f"{self.step_a_output}\n\n"

            "PREVIOUS OUTPUT (Step B):\n\n"

            f"{self.step_b_output}\n\n"

            "TASK:\n\n"

            "You are on Step 1: Figuring out what needs be added next to the subsection.\n\n"  

            "You have reasoned and then proposed mathematics instructions that are logically independent. " 
            "Since the bots will be working parallel, you thought about how any possible instructions depended on each other. "
            "You should not propose adding an instruction if you need to finish an earlier instruction first. "
            "Go through your previous output and create a list of all the instructions that you "
            "deemed independent and should be added to the current subsection.\n\n"

            "Examples of instruction X depending on instruction Y are:\n"
            "- If instruction X uses a definition from instruction Y.\n"
            "- If instruction X builds on examples introduced in instruction Y.\n"
            "- If instruction X references results from instruction Y.\n"
            "- If instruction X proves a conjecture from instruction Y."
            "- etc.\n\n"

            "Do NOT write the actual instructions here or work on any other steps."
        )

        self.prelim_reasoning_response = await llm_call(
            prompt=step_c_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L3_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L3 Bot Iteration #{self.iterations + 1} - Step C (Final reasoning output):\n" +
                "-" * 50 + "\n" +
                f"{self.prelim_reasoning_response}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _format_instructions_for_L4_bots(self):
        """LLM call to format instructions for L4 bots based on the math list, then sets up parallel tasks."""
        second_llm_prompt = (
            "DOCUMENT INSTRUCTIONS:\n\n"

            f"{self.L1_instruction}\n\n"

            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "SUBSECTION INSTRUCTIONS:\n\n"

            f"{self.L3_instruction}\n\n"

            "CURRENT SUBSECTION (working draft):\n\n"

            f"{self.subsection_draft}\n\n"

            "PREVIOUS REASONING (what we decided we need):\n\n"

            f"{self.reasoning_response}\n\n"

            "TASK:\n\n"

            "You are on Step 2: Writing instructions for the L4 bots on what math to construct.\n\n"

            "You need to convert the plan into instructions for the L4 bots. "
            "Each new or updated subsection should have its own instruction. Use the format:\n\n"

            "   INSTRUCTION 1\n"
            "   X\n"
            "   Y\n\n"

            "   INSTRUCTION 2\n"
            "   X\n"
            "   Y\n\n"

            "   etc.\n\n"

            "X is the title of the instruction: A few word summary of what the L4 bot will be doing.\n\n"

            "Y is text of the instruction: A detailed explanation of what the L4 bot will figure out. "
            "Be sure to include both what the bot should do and what it should not do (i.e., what is being "
            "delegated to other bots).\n\n "

            "Unless a result is well-known to the average graduate student, you should ALWAYS request a "
            "proof of a result. Avoid definitive language; for instance, it is better to say 'prove or " 
            "disprove X' instead of asserting the bot to 'prove X'. Never tell the bot to do anything " 
            "using numerical methods such as code (Python), WolframAlpha, OEIS, a scientific calculator etc."
        )

        self.formatted_instructions_response = await llm_call(
            prompt=second_llm_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L3_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L3 Bot Iteration #{self.iterations + 1} - Math instructions:\n" +
                "-" * 50 + "\n" +
                f"{self.formatted_instructions_response}\n" +
                "=" * 50 + "\n"
            )
            print(output)

        # Parse the formatted instructions into environment_instructions.
        self.environment_instructions = []
        lines = self.formatted_instructions_response.strip().splitlines()
        current_instruction = None
        has_started = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("INSTRUCTION "):
                has_started = True
                if current_instruction is not None:
                    self.environment_instructions.append("\n".join(current_instruction))
                current_instruction = []
            elif has_started:
                current_instruction.append(stripped_line)
        if current_instruction is not None:
            self.environment_instructions.append("\n".join(current_instruction))

        if Config.L3_PRINT:
            instructions_output = ""
            for i, inst in enumerate(self.environment_instructions):
                instructions_output += f"  Environment instruction #{i+1}:\n{inst}\n\n"

            output = (
                "\n" + "=" * 50 + "\n" +
                f"L3 Bot Iteration #{self.iterations + 1} - Parsed with instructions:\n" +
                "-" * 50 + "\n" +
                instructions_output +
                "=" * 50 + "\n"
            )
            print(output) 

        # Clear children and set up parallel tasks for L4 bots.
        self.children = [
            L4Bot(
                self.document, 
                self.section, 
                self.subsection_draft, 
                self.L1_instruction,
                self.L2_instruction, 
                self.L3_instruction, 
                instruction,
                self.lbl_mgr
            )
            for instruction in self.environment_instructions
            for _ in range(Config.NUM_L4_BOTS)
        ]        
        

    async def _draft_subsection_code(self):
        """LLM call to draft the subsection by integrating outputs from L4 bots."""
        # Group children by their L4_instruction
        children_by_instruction = defaultdict(list)
        for child in self.children:
            children_by_instruction[child.L4_instruction].append(child)

        # Create a new list containing one complete bot per instruction (if available)
        selected_children = []
        for instruction, bots in children_by_instruction.items():
            complete_bots = [bot for bot in bots if not bot.incomplete]
            if complete_bots:
                selected_children.append(random.choice(complete_bots))
            # If no complete bots exist for an instruction, that instruction group is omitted

        # Update self.children to only include the chosen complete bots
        self.children = selected_children

        # Check: if all math drafts are None, skip drafting.
        if len(self.children) == 0:
            if Config.L3_PRINT:
                output = (
                "\n" + "=" * 50 + "\n" +
                f"L3 Bot Iteration #{self.iterations + 1} - Draft subsection code:\n" +
                "-" * 50 + "\n" +
                f"No children are complete. Restarting...\n" +
                "=" * 50 + "\n"
            )
            print(output)
            self.restart = True
            return

        # Now build env_output based on the dictionary and track failed instructions
        env_output = ""
        idx = 1
        for instruction, draft in self.math_drafts.items():
            if draft is not None:
                env_output += (
                    f"----- Block {idx} -----\n{draft}\n\n"
                )
                idx += 1

        third_llm_prompt = (
            "DOCUMENT INSTRUCTIONS:\n\n"

            f"{self.L1_instruction}\n\n"

            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "SUBSECTION INSTRUCTIONS:\n\n"

            f"{self.L3_instruction}\n\n"

            "CURRENT SUBSECTION (working draft):\n\n"

            f"{self.subsection_draft}\n\n"

            "MATH BLOCKS (FROM L4 BOT):\n\n"

            f"{env_output}\n\n"

            "TASK:\n\n"

            r"You are on Step 3: Drafting a copy of the \subsection{...} by inserting the work of the L4 bots." "\n\n"

            "You need to write a new subsection draft to include the new mathematics. You should "
            "write like Serre and be economical: While you should show all necessary details, you "
            "should not include unnecessary or low-quality mathematics. In particular, if you find any "
            "of the math blocks repetitive or useless in nature, do not include them.\n\n" 
            
            "You are free to restructure the text to incorporate the math that you want in a coherent " 
            "manner, but you should not fundamentally change any mathematical statements to have a " 
            "different meaning. Only include mathematics from the blocks; do not derive any new results.\n\n"

            r"IMPORTANT: Always use proper LaTeX environments by using \begin{X}...\end{X} " 
            "where X can be theorem, proposition, lemma, example, proof, remark, definition, " 
            "corollary, etc. whenever possible. Each mathematical statement should be "
            "properly formatted in its corresponding environment. Avoid just writing math " 
            "in an expository manner without using environments.\n\n"

            r"Every theorem, proposition, lemma, example, proof, remark, "
            r"definition, corollary, etc. should have a label \label{Y}."
            "\n\n"

            "First reason about how the subsection should be structured. Then, when you are ready,"
            r"rite out the subsection draft, starting the first line of your response with \subsection{title}, "
            "where 'title' is what the subsection is named."
        )

        self.subsection_draft = await llm_call(
            prompt=third_llm_prompt,
            system_prompt=self.system_prompt
        )

        # Add custom labels
        pattern = r'\\(?:label|ref|eqref)\{([^}]+)\}'
        unique_labels = set(re.findall(pattern, self.subsection_draft))
        for label in unique_labels:
            if not await self.lbl_mgr.check_label(label):
                new_label = await self.lbl_mgr.get_label()
                self.subsection_draft = re.sub(
                    r'(\\(?:label|ref|eqref)\{' + re.escape(label) + r'\})',
                    lambda m: m.group(1)[:m.group(1).rfind(label)] + new_label + '}',
                    self.subsection_draft
                )

        if Config.L3_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L3 Bot Iteration #{self.iterations + 1} - Draft subsection code:\n" +
                "-" * 50 + "\n" +
                f"{self.subsection_draft}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _final_decision_for_subsection(self):
        """LLM call to decide if the drafted subsection is complete or needs further refinement."""
        final_llm_prompt = (
            "DOCUMENT INSTRUCTIONS:\n\n"

            f"{self.L1_instruction}\n\n"

            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION:\n\n"

            f"{self.section}\n\n"

            "SUBSECTION INSTRUCTIONS:\n\n"

            f"{self.L3_instruction}\n\n"

            "CURRENT SUBSECTION (working draft):\n\n"

            f"{self.subsection_draft}\n\n"

            "TASK:\n\n"

            r"You are on Step 4: Deciding whether the \subsection{...} is in satisfactory " 
            "shape or needs to go through further revisions.\n\n"

            "First, discuss whether you think this is a satisfactory subsection given the "
            "instructions for it. Then, if it is complete, respond on the final line with 'COMPLETE', " 
            "else write 'REFINE'.\n\n"

            "Assume that any and all of the mathematics presented is indeed " 
            "correct."
        )

        self.final_decision_response = await llm_call(
            prompt=final_llm_prompt,
            system_prompt=self.system_prompt
        )
        self.final_decision_response = self.final_decision_response.strip()

        if Config.L3_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L3 Bot Iteration #{self.iterations + 1} - Final decision:\n" +
                "-" * 50 + "\n" +
                f"{self.final_decision_response}\n" +
                "=" * 50 + "\n"
            )
            print(output)

        self.iterations += 1

        # If the evaluation returns COMPLETE or if we've reached the iteration limit, mark as done.
        final_line = self.final_decision_response.splitlines()[-1].strip().upper()
        if ("COMPLETE" in final_line) or (not self.iterations < Config.L3_REASONING_STEPS):
            self.done = True


    async def step(self):
        """
        Execute the next LLM call in the sequence.
        Each call to this method triggers exactly one llm_call.
        """
        if self.done:
            raise RuntimeError("L3 bot called after being marked done.")
        if not self.iterations < Config.L3_REASONING_STEPS:
            raise RuntimeError("Iteration limit reached: Maximum number of L3 reasoning steps exceeded.")

        # Build the llm call sequence up to formatting instructions.
        llm_call_sequence = [
            self._reasoning_step_A,
            self._reasoning_step_B,
            self._reasoning_step_C,
            self._format_instructions_for_L4_bots,
            self._draft_subsection_code,
            self._final_decision_for_subsection
        ]

        # Execute the current llm call.
        await llm_call_sequence[self.current_llm_call_index]()

        if self.restart:
            self.current_llm_call_index = 0
            self.restart = False
        else:
            self.current_llm_call_index = (self.current_llm_call_index + 1) % len(llm_call_sequence)
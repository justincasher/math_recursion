# bots/L3_bot.py

import asyncio
from collections import defaultdict
import random
import re

from bots.L4_bot import L4Bot
from config import Config
from latex_labels import LabelManager
from llm_call import llm_call
from round_robin.L4_round_robin import L4RoundRobin


class L3Bot:
    def __init__(
            self, 
            document: str, 
            section: str, 
            subsection_draft: str,
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


    async def _generate_math_list(self):
        """LLM call to generate a list of math to add based on the preliminary reasoning."""
        first_llm_prompt = (
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

            "REASONING:\n\n"

            f"{self.prelim_reasoning_response}\n\n"

            "TASK:\n\n"

            "You are still on Step 1: Figuring out what math need to be added next to the current subsection.\n\n"

            "You just reasoned about what math you need to add to the subsection next (see REASONING). "
            "Since the bots will be working parallel, you thought about how any possible instructions depended on each other. "
            "You should not propose adding an instruction if you need to finish an earlier instruction first. "

            "Examples of instruction X depending on instruction Y are:\n"
            "- If instruction X uses a definition from instruction Y.\n"
            "- If instruction X builds on examples introduced in instruction Y.\n"
            "- If instruction X references results from instruction Y.\n"
            "- If instruction X proves a conjecture from instruction Y."
            "- etc.\n\n"

            "Now, write a list of what math you concluded from your reasoning that you should write next. "
            "Make sure to say what type of math you want (theorem, proposition, lemma, example, conjecture, etc.), " 
            "along with a description of what the math should contain. Also, be clear about what it should NOT contain "
            "(i.e., what is being delegated to the other bots)."
        )

        self.reasoning_response = await llm_call(
            prompt=first_llm_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L3_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L3 Bot Iteration #{self.iterations + 1} - List of new math:\n" +
                "-" * 50 + "\n" +
                f"{self.reasoning_response}\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _format_instructions_for_L4_bots(self):
        """LLM call to format instructions for L4 bots based on the math list, then sets up parallel tasks."""
        second_llm_prompt = (
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
            "delegated to other bots). If requested result is not very well known, you should require the bot to " 
            "produce a proof."
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
                self.L2_instruction, 
                self.L3_instruction, 
                instruction,
                self.lbl_mgr
            )
            for instruction in self.environment_instructions
            for _ in range(Config.NUM_L4_BOTS)
        ]

    
    async def _create_round_robin(self):
        """
        Create round robin tournaments for each instruction.
        This method groups the L4 bots (from self.children) by their L4_instruction,
        filters out those that are incomplete, and then creates pairwise L4RoundRobin
        instances for each group with more than one complete bot.
        """
        # Group complete L4 bots by their instruction.
        instruction_groups = defaultdict(list)
        for child in self.children:
            if not child.incomplete:
                instruction_groups[child.L4_instruction].append(child)

        # For each instruction group, create pairwise round robin matches.
        for instruction, bots in instruction_groups.items():
            if len(bots) > 1:
                # Compare each pair only once.
                for i in range(len(bots)):
                    for j in range(i + 1, len(bots)):
                        rr_match = L4RoundRobin(bots[i], bots[j])
                        self.round_robin.append(rr_match)

    
    def _compute_round_robin_winners(self):
        """
        Compute the Elo ratings for each L4 bot based on the round robin tournament results,
        and update self.children to retain only the winning bots for each instruction.
        """
        K = 32  # Elo update factor

        # Group L4 bots by their instruction (all bots regardless of completeness).
        groups = defaultdict(list)
        for bot in self.children:
            groups[bot.L4_instruction].append(bot)

        # Initialize Elo ratings for each bot.
        ratings = {bot: 1500 for bots in groups.values() for bot in bots}

        # Update Elo ratings based on round robin matches.
        for match in self.round_robin:
            # Only process matches where both bots have ratings.
            if match.bot1 in ratings and match.bot2 in ratings:
                r1, r2 = ratings[match.bot1], ratings[match.bot2]
                # Determine scores: 1 for win, 0 for loss, 0.5 for tie.
                if match.result == 1:
                    s1, s2 = 1, 0
                elif match.result == 2:
                    s1, s2 = 0, 1
                else:
                    s1, s2 = 0.5, 0.5
                e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
                ratings[match.bot1] = r1 + K * (s1 - e1)
                ratings[match.bot2] = r2 + K * (s2 - e2)

        # Determine the winning bot for each instruction group.
        winners = {}
        for instr, bots in groups.items():
            # Filter for complete bots.
            complete_bots = [bot for bot in bots if not bot.incomplete]
            if complete_bots:
                winners[instr] = max(complete_bots, key=lambda b: ratings.get(b, 1500))
            else:
                # If no bots are complete, choose one randomly among all bots for this instruction.
                winners[instr] = random.choice(bots)

        # Update self.children: retain only the winning bot for each instruction.
        self.children = list(winners.values())
        

    async def _draft_subsection_code(self):
        """LLM call to draft the subsection by integrating outputs from L4 bots."""
        # Start by figuring out who wone the round robing tournament if 
        # there was one
        if Config.NUM_L4_BOTS > 1:
            self._compute_round_robin_winners()

        # Populate self.math_drafts with instructions and corresponding math drafts (or None)
        for child in self.children:
            if not child.incomplete:
                self.math_drafts[child.L4_instruction] = child.math_draft
            else:
                self.math_drafts[child.L4_instruction] = None

        # Check: if all math drafts are None, skip drafting.
        if all(draft is None for draft in self.math_drafts.values()):
            if Config.L3_PRINT:
                print("All math drafts are None. Restarting llm_call_sequence.")
            self.restart = True
            return None

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
            "of the math blocks repetitive or useless in nature, do not include them. You are free to restructure "
            "the text to incorporate the math that you want in a coherent manner, but you should not fundamentally "
            "change any mathematical statements to have a different meaning.\n\n"

            r"IMPORTANT: Always use proper LaTeX environments by using \begin{X}...\end{X} " 
            "where X can be theorem, proposition, lemma, example, proof, remark, definition, " 
            "corollary, etc. whenever possible. Each mathematical statement should be "
            "properly formatted in its corresponding environment. Avoid just writing math " 
            "in an expository manner without using environments.\n\n"

            r"Every theorem, proposition, lemma, example, proof, remark, "
            r"definition, corollary, etc. should have a label \label{Y}."
            "\n\n"

            r"ONLY write out the subsection draft, starting the first line of your response with \subsection{title}, "
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
        if self.iterations >= Config.L3_REASONING_STEPS:
            raise RuntimeError("Iteration limit reached: Maximum number of L3 reasoning steps exceeded.")

        # Build the llm call sequence up to formatting instructions.
        llm_call_sequence = [
            self._reasoning_step_A,
            self._reasoning_step_B,
            self._reasoning_step_C,
            self._generate_math_list,
            self._format_instructions_for_L4_bots,
        ]

        # If multiple L4 bots are configured, create round robin tournaments after formatting instructions.
        if Config.NUM_L4_BOTS > 1:
            llm_call_sequence.extend([
                self._create_round_robin,
            ])

        # Append the remaining steps.
        llm_call_sequence.extend([
            self._draft_subsection_code,
            self._final_decision_for_subsection,
        ])

        # Execute the current llm call.
        await llm_call_sequence[self.current_llm_call_index]()

        if self.restart:
            self.current_llm_call_index = 0
            self.restart = False
        else:
            self.current_llm_call_index = (self.current_llm_call_index + 1) % len(llm_call_sequence)
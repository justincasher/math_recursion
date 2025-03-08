# bots/L2_bot.py

import asyncio
from collections import defaultdict
import random
import re

from bots.L3_bot import L3Bot
from config import Config
from latex_labels import LabelManager
from llm_call import llm_call
from round_robin.L3_round_robin import L3RoundRobin


class L2Bot:
    def __init__(
            self, 
            document: str,
            L2_instruction: str, 
            section_draft: str,
            lbl_mgr: LabelManager
        ):
        """
        Initialize the L2Bot instance.

        This bot is responsible for constructing a specific LaTeX section based on 
        instructions from an L1 bot. It determines necessary subsections, delegates 
        their creation to L3 bots, and iteratively refines the section until it 
        reaches a satisfactory state.
        """
        self.document = document
        self.L2_instruction = L2_instruction
        self.section_draft = section_draft
        self.lbl_mgr = lbl_mgr

        self.system_prompt = (
            "You are a Level 2 (L2) bot. You are writing a LaTeX document with other bots, and you are " 
            r"responsible for a single \section{...}. You received instructions on what this section should contain "  
            "from an L1 bot. You will delegate any subsection-level work to an L3 bot. "
            "These L3 bots will work in *parallel*, meaning you should not propose changes that depend on eachother.\n\n"

            "You will iterate through a multi-step process composed of the following:\n"
            r"1. Figuring out how the current section can be improved." "\n"
            r"2. Writing instructions for the L3 bots on how to write these subsections in parallel." "\n"
            r"3. Drafting a copy of the \section{...} by inserting the subsection work of the L3 bots." "\n"
            r"4. Deciding whether the \section{...} is in satisfactory shape or needs to go through further revisions." "\n\n"

            "Notes:\n"
            "- Be economical about what you write, always considering how it relates to the instructions.\n"
            "- Use LaTeX when writing math, but NEVER write out an entire document, just the relevant text.\n"
            r"- Use $...$ and $$...$$ instead of \(...\) and \[...\]."
            "- Use LaTeX environments like gather, theorem, align, lemma, proof, example, etc.\n"
            "- Do NOT attempt or respond about any other steps than the one your are on."
            "- Never use numerical tools (i.e., methods) such as code (Python), WolframAlpha, OEIS, etc."
        )

        # Variables for pre-parallel processing
        self.prelim_reasoning_response = "N/A"
        self.reasoning_response = "N/A"
        self.formatted_instructions_response = "N/A"
        self.L3_instructions = []
        self.children = []  # List of L3Bot instances

        # Variables for post-parallel processing
        self.final_decision_response = "N/A"

        # Control attributes
        self.iterations = 0
        self.done = False

        # Tracks which LLM call to perform next
        self.current_llm_call_index = 0

        # The produced subsections
        self.subsection_blocks = self._extract_subsection_blocks()


    def _extract_subsection_blocks(self):
            """
            Extracts subsection blocks from the section draft.
            Returns a dictionary mapping each subsection title to its corresponding block.
            """
            pattern = r'(\\subsection\{[^}]+\})(.*?)(?=(\\subsection\{|$))'
            matches = re.findall(pattern, self.section_draft, re.DOTALL)
            blocks = {}
            for header, content, _ in matches:
                block = (header + content).rstrip()
                title_match = re.search(r'\\subsection\{([^}]+)\}', block)
                if title_match:
                    title = title_match.group(1).strip()
                    blocks[title] = block
            return blocks


    async def _preliminary_reasoning(self):
        """LLM call to reason about which subsections could be added next."""
        reasoning_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS:\n\n"

            f"{self.L2_instruction}\n\n"
            
            "CURRENT SECTION (working draft):\n\n"

            f"{self.section_draft}\n\n"

            "REASON FOR REFINEMENT:\n\n"

            f"{self.final_decision_response}\n\n"

            "TASK:\n\n"

            "You are on Step 1: Figuring out how the current section can be improved.\n\n"

            "You need to reason about what new subsections you can add or existing subsections " 
            "you could improve (edit). Note that document components like introductory paragraphs, " 
            "summaries, and transitions are NOT considered subsections - they are structural elements. You should ONLY propose "
            "adding a subsection if it is logically independent of other proposed subsections. This means that it does "
            "not utilize or redevelop any content developed in the other proposed subsections. It is important to be "
            "efficient: If subsection X might use material from subsection Y, first write subsection Y.\n\n"

            "Examples of subsection X depending on subsection Y are:\n"
            "- If subsection X uses a definition from subsection Y.\n"
            "- If subsection X builds on examples introduced in subsection Y.\n"
            "- If subsection X references results from subsection Y\n"
            "- etc.\n\n"

            "Do the following:\n"
            "A) Reason about what content subsections could be added or improved, detailing what is in each subsection.\n"
            "B) Reason step-by-step about how your proposed subsections depend on each other.\n"
            "C) Consider potential improvements for existing subsections.\n"
            "D) Double check that your reasoning makes sense.\n\n"

            "Notes:\n"
            "- Only complete A)–D); do NOT write anything else.\n"
            "- It is best to be stringent when proposing subsections.\n"
            "- Do NOT provide final proposals or list concrete changes; use this space as a scratchpad for ideas.\n"
            "- Do NOT write any instructions here or work on any other steps.\n"
        )

        self.prelim_reasoning_response = await llm_call(
            prompt=reasoning_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L2_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L2 Bot Iteration #{self.iterations + 1} - Preliminary Reasoning\n" +
                "-" * 50 + "\n" +
                self.prelim_reasoning_response + "\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _generate_subsection_list(self):
        """LLM call to produce a list of subsections to add, based on prior reasoning."""
        first_llm_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS:\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION (working draft):\n\n"

            f"{self.section_draft}\n\n"

            "REASONING:\n\n"

            f"{self.prelim_reasoning_response}\n\n"

            "TASK:\n\n"

            "You are still on Step 1: Figuring out how the current section can be improved.\n\n"

            "You just reasoned about what subsections you need to add to the section next (see REASONING). "
            "Since the bots will be working parallel, you thought about how any possible subsections depended on each other. "
            "You should not propose adding a subsection if you need to finish earlier sections first. For instance, " 
            "it would be illogical to propose adding a subsection to a Real Analysis textbook on Cauchy sequences at " 
            "the same time as a subsection on defining what a sequence is, because the definition of a Cauchy sequence " 
            "depends on the definition of a sequence.\n\n"
 
            "Write a list of what subsections you will add immediately to the document, based on the conclusions "
            "of your reasoning. Make sure to give the title of each subsection along with a description of what " 
            "that subsection should contain. Also, be clear about what it should NOT contain (i.e, "
            "what is being delegated to the other bots)."
        )

        self.reasoning_response = await llm_call(
            prompt=first_llm_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L2_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L2 Bot Iteration #{self.iterations + 1} - List of Subsections\n" +
                "-" * 50 + "\n" +
                self.reasoning_response + "\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _format_instructions_for_L3_bots(self):
        """LLM call to format instructions for L3 bots to generate the subsections."""
        second_llm_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS:\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION (working draft):\n\n"

            f"{self.section_draft}\n\n"

            "PREVIOUS REASONING (what we decided we need):\n\n"

            f"{self.reasoning_response}\n\n"

            "TASK:\n\n"

            "You are on Step 2: Writing instructions for the L3 bots on how to write these subsections.\n\n"

            "You need to convert the plan into instructions for the L3 bots."
            "Each new or updated subsection should have its own instruction. Use the format:\n\n"

            "   INSTRUCTION 1\n"
            "   X\n"
            "   Y\n\n"

            "   INSTRUCTION 2\n"
            "   X\n"
            "   Y\n\n"

            "   etc.\n\n"

            "X is the title of the subsection that you would like to create or edit.\n\n"

            "If you would like to edit a subsection, its title should be an exact match of the value X "
            r"where the subsection that you would like to edit is \subsection{X}. "
            "If you are creating a new subsection, its title should be descriptive.\n\n"
            
            "Y is the text of the instruction: What should be done in that subsection."
            "Be sure to include both what the bot should do and what it should not do (i.e., what is being "
            "delegated to other bots).\n\n"

            "You should take a high-level approach when describing what the subsection contains. "
            "Asserting that the subsection MUST contain a certain result which is, in fact, infeasible, "
            "or must prove a result via a certain method that does not work, will derail the bots. "
            "You can make suggestions, but try to avoid definitive language."
        )

        self.formatted_instructions_response = await llm_call(
            prompt=second_llm_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L2_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L2 Bot Iteration #{self.iterations + 1} - Subsection Instructions\n" +
                "-" * 50 + "\n" +
                self.formatted_instructions_response + "\n" +
                "=" * 50 + "\n"
            )
            print(output)

        # Parse the formatted instructions into L3_instructions.
        self.L3_instructions = []
        lines = self.formatted_instructions_response.strip().splitlines()
        current_instruction = None
        has_started = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("INSTRUCTION "):
                has_started = True
                if current_instruction is not None:
                    self.L3_instructions.append("\n".join(current_instruction))
                current_instruction = []
            elif has_started:
                current_instruction.append(stripped_line)

        if current_instruction is not None:
            self.L3_instructions.append("\n".join(current_instruction))

        if Config.L2_PRINT:
            instructions_output = ""
            for i, inst in enumerate(self.L3_instructions):
                instructions_output += f"  Sub-instruction #{i+1}:\n{inst}\n\n"
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L2 Bot Iteration #{self.iterations + 1} - Parsed Subsection Instructions\n" +
                "-" * 50 + "\n" +
                instructions_output +
                "=" * 50 + "\n"
            )
            print(output)

        # Instantiate L3Bot children for parallel execution.
        self.children = []
        for L3_instruction in self.L3_instructions:
            lines = [line.strip() for line in L3_instruction.splitlines() if line.strip()]
            subsection_title = lines[0] if lines else ""
            subsection_draft = self.subsection_blocks.get(subsection_title, "N/A")
            for _ in range(Config.NUM_L3_BOTS):
                self.children.append(
                    L3Bot(
                        self.document, 
                        self.section_draft, 
                        subsection_draft,
                        self.L2_instruction, 
                        L3_instruction,
                        self.lbl_mgr
                    )
                )


    async def _create_round_robin(self):
        """
        Create round robin tournaments for each instruction.
        This method groups the L3 bots (from self.children) by their L3_instruction
        and then creates pairwise L3RoundRobin instances for each group with 
        more than one complete bot.
        """
        self.round_robin = []
        instruction_groups = defaultdict(list)
        for child in self.children:
            instruction_groups[child.L3_instruction].append(child)

        for instruction, bots in instruction_groups.items():
            if len(bots) > 1:
                for i in range(len(bots)):
                    for j in range(i + 1, len(bots)):
                        rr_match = L3RoundRobin(bots[i], bots[j])
                        self.round_robin.append(rr_match)


    def _compute_round_robin_winners(self):
        """
        Compute the Elo ratings for each L3 bot based on the round robin tournament results,
        and update self.children to retain only the winning bots for each instruction.
        """
        K = 32  # Elo update factor

        # Group L3 bots by their instruction (all bots regardless of completeness).
        groups = defaultdict(list)
        for bot in self.children:
            groups[bot.L3_instruction].append(bot)

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
            winners[instr] = max(bots, key=lambda b: ratings.get(b, 1500))

        # Update self.children: retain only the winning bot for each instruction.
        self.children = list(winners.values())


    async def _draft_section_code(self):
        """LLM call to draft the section by integrating L3 bot outputs as placeholders."""
        # Start by figuring out who wone the round robing tournament if 
        # there was one
        if Config.NUM_L3_BOTS > 1:
            self._compute_round_robin_winners()

        # Populate self.subsection_blocks with titles and corresponding subsection drafts (or None)
        for child in self.children:
            if child.subsection_draft is not None:
                sub_title_match = re.search(r'\\subsection\{([^}]+)\}', child.subsection_draft)
                if sub_title_match:
                    title = sub_title_match.group(1).strip()
                    self.subsection_blocks[title] = child.subsection_draft

        # Now build subsection_output based on the dictionary and track failed instructions
        subsection_output = ""
        idx = 1
        for title, draft in self.subsection_blocks.items():
            if draft is not None:
                subsection_output += (
                    f"----- Subsection {idx} ({title}) -----\n{draft}\n\n"
                )
                idx += 1


        third_llm_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "SECTION INSTRUCTIONS:\n\n"

            f"{self.L2_instruction}\n\n"

            "CURRENT SECTION (working draft):\n\n"

            f"{self.section_draft}\n\n"

            "SUBSECTIONS (from L3 bots):\n\n"

            f"{subsection_output}\n\n"

            "TASK:\n\n"

            r"You are on Step 3: Drafting a copy of the \section{...} by inserting the subsection work of the L3 bots." "\n\n"

            "You need to update the current section draft to include the new subsections. Note that, while "
            "you can add introductory paragraphs, explaining what this section will cover, you cannot write any "
            "NEW subsections here. Most of the mathematics will be copy and pasted from the provided subsections.\n\n"

            r"You can insert a subsection so by writing '\subsection{X}' on a line by itself, where X is the name of the "
            "subsection that you would like to go there. This will paste the ENTIRE subsection there once "
            "you are done writing. You do not need to write ANY of the content of the subsection besides its title.\n\n"

            r"ONLY write out the section draft, starting the first line of your response with \section{title}, "
            "where 'title' is what the section is named."
        )

        self.draft_section_code = await llm_call(
            prompt=third_llm_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L2_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L2 Bot Iteration #{self.iterations + 1} - Draft Section Code\n" +
                "-" * 50 + "\n" +
                self.draft_section_code + "\n" +
                "=" * 50 + "\n"
            )
            print(output)

        # Identify the subsection titles actually used in the draft section code.
        used_subtitles = set(re.findall(r'\\subsection\{([^}]+)\}', self.draft_section_code))

        # Discard subsections that are not referenced in the draft section code.
        self.subsection_blocks = {title: block for title, block in self.subsection_blocks.items() if title in used_subtitles}

        # Replace subsection placeholders with the actual subsection content.
        def replace_subsection(match_obj):
            title = match_obj.group(1).strip()
            return self.subsection_blocks.get(title, match_obj.group(0))

        self.section_draft = re.sub(
            r'\\subsection\{([^}]+)\}', 
            replace_subsection, 
            self.draft_section_code
        )

        if Config.L2_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L2 Bot Iteration #{self.iterations + 1} - Section Draft\n" +
                "-" * 50 + "\n" +
                self.section_draft + "\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _final_decision_for_section(self):
        """LLM call to decide if the current section draft is complete or needs further refinement."""
        final_llm_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document}\n\n"

            "CURRENT SECTION (working draft):\n\n"

            f"{self.section_draft}\n\n"

            "INSTRUCTIONS FROM L1 BOT:\n\n"

            f"{self.L2_instruction}\n\n"

            "DRAFT SECTION CODE:\n\n"

            f"{self.section_draft}\n\n"

            "TASK:\n\n"

            r"You are on Step 4: Deciding whether the \section{...} is in satisfactory " 
            "shape or needs to go through further revisions.\n\n"
            
            "You have produced the given section above. You have the option to keep editing it "
            "and adding new subsection, or mark the section as complete. It should be relatively clear "
            "whether or not the current section can be improved upon, or if it is in satisfactory "
            "shape given the instructions for what it should contain.\n\n"

            "First, reason extensively as to whether you think this is a satisfactory document given the "
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

        if Config.L2_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L2 Bot Iteration #{self.iterations + 1} - Final Decision\n" +
                "-" * 50 + "\n" +
                self.final_decision_response + "\n" +
                "=" * 50 + "\n"
            )
            print(output)

        self.iterations += 1

        # If the evaluation returns COMPLETE or if we've reached the iteration limit, mark as done.
        final_line = self.final_decision_response.splitlines()[-1].strip().upper()
        if ("COMPLETE" in final_line) or (not self.iterations < Config.L2_REASONING_STEPS):
            self.done = True


    async def step(self):
        """
        Execute the next LLM call in the sequence.
        Each invocation of this method results in exactly one llm_call.
        """
        if self.done:
            raise RuntimeError("L2 bot called after being marked done.")
        if self.iterations >= Config.L2_REASONING_STEPS:
            raise RuntimeError("Iteration limit reached: Maximum number of L2 reasoning steps exceeded.")

        # Build the llm call sequence.
        llm_call_sequence = [
            self._preliminary_reasoning,
            self._generate_subsection_list,
            self._format_instructions_for_L3_bots,
        ]

        # If multiple L3 bots are configured, incorporate round robin tournaments.
        if Config.NUM_L3_BOTS > 1:
            llm_call_sequence.extend([
                self._create_round_robin,
            ])

        # Append the remaining steps.
        llm_call_sequence.extend([
            self._draft_section_code,
            self._final_decision_for_section,
        ])

        # Execute the current LLM call.
        await llm_call_sequence[self.current_llm_call_index]()

        # Move to the next step in the sequence.
        self.current_llm_call_index += 1
        if self.current_llm_call_index >= len(llm_call_sequence):
            self.current_llm_call_index = 0

# bots/L1_bot.py

import asyncio
from collections import defaultdict
import random
import re

from bots.L2_bot import L2Bot
from config import Config
from latex_labels import LabelManager
from llm_call import llm_call
from round_robin.L2_round_robin import L2RoundRobin


class L1Bot:
    def __init__(
            self, 
            document_draft: str, 
            lbl_mgr: LabelManager
        ):
        """
        Initialize the L1Bot instance.

        This bot is responsible for the high-level construction of a LaTeX document.
        It determines the structure of the document, delegates section-level writing
        tasks to L2 bots, and iteratively refines the document until it reaches a 
        satisfactory state.
        """
        self.system_prompt = (
            "You are a Level 1 (L1) bot. You are writing a LaTeX document with other bots, and you are "
            "responsible for the high-level construction of the LaTeX document. You can modify the document "
            "preamble, create, edit, or remove sections. You will delegate any section-level work to an L2 bot. "
            "These L2 bots will work in *parallel*, meaning you should not propose changes that depend on eachother.\n\n"

            "You will iterate through a multi-step process composed of the following:\n"
            "1. Figuring out what changes need to be be made next to the document.\n"
            "2. Writing instructions for the L2 bots on how to write these sections in parallel.\n"
            "3. Drafting a copy of the document by inserting the work of the L2 bots.\n"
            "4. Deciding whether the document is in satisfactory shape or needs to go through further revisions.\n\n"

            "Notes:\n"
            "- Be economical about what you write, always considering how it relates to the goal of the paper.\n"
            "- Use LaTeX when writing math.\n"
            r"- Use $...$ and $$...$$ instead of \(...\) and \[...\]."
            "- Use LaTeX environments like gather, theorem, align, lemma, proof, example, etc.\n"
            "- Do NOT attempt or respond about any other steps than the one your are on."
            "- Never use numerical tools (i.e., methods) such as code (Python), WolframAlpha, OEIS, etc."
        )

        # Primary document state and iteration details
        self.document_draft = document_draft
        self.lbl_mgr = lbl_mgr

        # Variables used in pre-parallel processing
        self.prelim_reasoning_response = ""
        self.reasoning_response = ""
        self.formatted_instructions_response = ""
        self.L2_instructions = []
        self.children = []  # List of L2Bot instances

        # Variables used in post-parallel processing
        self.draft_document_code = ""
        self.final_decision_response = "N/A"

        # Control attributes for iterations
        self.iterations = 0
        self.done = False

        # Index for LLM call sequence within one iteration.
        self.current_llm_call_index = 0

        # The produced sections
        self.section_blocks = self._extract_section_blocks()


    def _extract_section_blocks(self):
        """
        Extracts section blocks from the document_draft.
        Returns a dictionary mapping each section title to its corresponding block.
        """
        pattern = r'(\\section\{[^}]+\})(.*?)(?=(\\section\{|\\end\{document\}|$))'
        matches = re.findall(pattern, self.document_draft, re.DOTALL)
        blocks = {}
        for header, content, _ in matches:
            block = (header + content).rstrip()
            sec_title_match = re.search(r'\\section\{([^}]+)\}', block)
            if sec_title_match:
                title = sec_title_match.group(1).strip()
                blocks[title] = block
        return blocks


    async def _preliminary_reasoning(self):
        """LLM call to reason about which sections could be added next."""
        reasoning_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document_draft}\n\n"

            "REASON FOR REFINEMENT:\n\n"

            f"{self.final_decision_response}\n\n"

            "TASK:\n\n"

            "You are completing Step 1: Figuring out what changes need to be made next to the document.\n\n"

            "You need to reason about what new content sections you can add given the document in parallel. " 
            "Note that document components like the abstract, acknowledgments, and references are NOT " 
            "considered sections - they are special components with different purposes. You should ONLY propose "
            "adding a section if it is logically independent of the other proposed sections. This means that it does "
            "not utilize or redevelop any theory developed in the other proposed sections. It is important to be "
            "efficient: If section X might use theory from section Y, first write section Y.\n\n"

            "Examples of section X depending on section Y are:\n"
            "- If section X uses a definition from section Y.\n"
            "- If section X computes examples which can be used to write section Y.\n"
            "- If section X uses cites a theorem from section Y\n"
            "- etc.\n\n"

            "Do the following:\n"
            "A) Reason about what content sections could be added, detailing what is in each section.\n"
            "B) Reason step-by-step about how your proposed sections depend on each other.\n"
            "C) Propose sections which are logically independent and should be written next.\n"
            "D) Double check that your answer makes sense.\n\n"

            "Notes:\n"
            "- Only complete A)–D); do NOT write anything else.\n"
            "- It is best to be stringent when proposing sections.\n"
            "- Focus only on true content sections, not document components like abstract, introduction, etc.\n"
        )
        
        self.prelim_reasoning_response = await llm_call(
            prompt=reasoning_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L1_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L1 Bot Iteration #{self.iterations + 1} - Preliminary Reasoning\n" +
                "-" * 50 + "\n" +
                self.prelim_reasoning_response + "\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _section_reasoning_list(self):
        """LLM call to produce a list of sections to be added immediately based on prior reasoning."""
        first_llm_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document_draft}\n\n"

            "REASONING:\n\n"

            f"{self.prelim_reasoning_response}\n\n"

            "TASK:\n\n"

            "You are still on Step 1: Figuring out what sections need to be added next to the document.\n\n"

            "You just reasoned about what sections you need to add to the document next (see REASONING). "
            "Since the bots will be working parallel, you thought about how any possible sections depended on eachother. "
            "You should not propose adding a section if you need to finish earlier sections first. For instance, it would "
            "be illogical to propose adding a section to a Real Analysis textbook on limits at the same time you are adding a "
            "section on sequences, since limits need sequences in order to be defined.\n\n"

            "Write a list of what sections you will added immediately to the document, based on the conclusions "
            "of your reasoning. Make sure to give the title of each section along with a description of what "
            "that section should contain. Also, be clear about what it should NOT contain (i.e, "
            "what is being delegated to the other bots)."
        )

        self.reasoning_response = await llm_call(
            prompt=first_llm_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L1_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L1 Bot Iteration #{self.iterations + 1} - Section Reasoning List\n" +
                "-" * 50 + "\n" +
                self.reasoning_response + "\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _format_instructions_for_L2(self):
        """LLM call to convert the section reasoning into formatted instructions for L2 bots."""
        second_llm_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document_draft}\n\n"

            "PLAN:\n\n"

            f"{self.reasoning_response}\n\n"

            "TASK:\n\n"

            "You are on Step 2: Writing instructions for the L2 bots on how to write these sections.\n\n"

            "You need to convert the plan into instructions for the L2 bots."
            "Each new or updated section should have its own title and instruction. Use the format:\n\n"

            "   INSTRUCTION 1\n"
            "   X\n"
            "   Y\n\n"

            "   INSTRUCTION 2\n"
            "   X\n"
            "   Y\n\n"

            "   etc.\n\n"

            "X is the title of the section that you would like to create or edit.\n\n"

            "If you would like to edit a section, its title should be an exact match of the value X "
            r"where the section that you would like to edit is \section{X}. "
            "If you are creating a new section, its title should be descriptive.\n\n"

            "Y is the text of the instruction is what should be done in that section."
            "Be sure to include both what the bot should do and what it should not do (i.e., what is being "
            "delegated to other bots).\n\n"

            "You should take a high-level approach when describing what the section contains. "
            "Asserting that it MUST contain a certain result which is, in fact, infeasible, "
            "or must prove a result via a certain method that does not work, will derail the bots. "
            "You can make suggestions, but try to avoid definitive language."
        )

        self.formatted_instructions_response = await llm_call(
            prompt=second_llm_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L1_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L1 Bot Iteration #{self.iterations + 1} - Format Instructions for L2 Bots\n" +
                "-" * 50 + "\n" +
                self.formatted_instructions_response + "\n" +
                "=" * 50 + "\n"
            )
            print(output)

        # Parse the instructions into a list of L2_instructions
        self.L2_instructions = []
        lines = self.formatted_instructions_response.strip().splitlines()
        current_instruction = []
        has_started = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("INSTRUCTION "):
                has_started = True
                if current_instruction:
                    self.L2_instructions.append("\n".join(current_instruction))
                    current_instruction = []
                continue
            if not has_started:
                continue
            current_instruction.append(stripped_line)

        if current_instruction:
            self.L2_instructions.append("\n".join(current_instruction))

        # Save the tasks for parallel execution by instantiating L2Bot children.
        self.children = []
        for L2_instruction in self.L2_instructions:
            lines = [line.strip() for line in L2_instruction.splitlines() if line.strip()]
            section_title = lines[0] if lines else ""
            section_draft = self.section_blocks.get(section_title, "N/A")
            for _ in range(Config.NUM_L2_BOTS):
                self.children.append(
                    L2Bot(
                        self.document_draft, 
                        L2_instruction, 
                        section_draft,
                        self.lbl_mgr
                    )
                )

    
    async def _create_round_robin(self):
        """
        Create round robin tournaments for each instruction.
        This method groups the L2 bots (from self.children) by their L2_instruction
        and then creates pairwise L2RoundRobin instances for each group with more 
        than one complete bot.
        """
        self.round_robin = []
        instruction_groups = defaultdict(list)
        for child in self.children:
            instruction_groups[child.L2_instruction].append(child)

        for instruction, bots in instruction_groups.items():
            if len(bots) > 1:
                for i in range(len(bots)):
                    for j in range(i + 1, len(bots)):
                        rr_match = L2RoundRobin(bots[i], bots[j])
                        self.round_robin.append(rr_match)


    def _compute_round_robin_winners(self):
        """
        Compute the Elo ratings for each L2 bot based on the round robin tournament results,
        and update self.children to retain only the winning bots for each instruction.
        """
        K = 32  # Elo update factor

        # Group L2 bots by their instruction (all bots regardless of completeness).
        groups = defaultdict(list)
        for bot in self.children:
            groups[bot.L2_instruction].append(bot)

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


    async def _draft_document_code(self):
        """LLM call to integrate L2 bots’ outputs and produce a new LaTeX document draft with placeholders."""
        # Start by figuring out who wone the round robing tournament if 
        # there was one
        if Config.NUM_L2_BOTS > 1:
            self._compute_round_robin_winners()

        # Populate self.section_blocks with section titles and corresponding section drafts (or None)
        for child in self.children:
            if child.section_draft is not None:
                sec_title_match = re.search(r'\\section\{([^}]+)\}', child.section_draft)
                if sec_title_match:
                    title = sec_title_match.group(1).strip()
                    self.section_blocks[title] = child.section_draft

        # Now build section_output based on the dictionary and track failed instructions
        section_output = ""
        idx = 1
        for title, draft in self.section_blocks.items():
            if draft is not None:
                section_output += (
                    f"----- Section {idx} ({title}) -----\n{draft}\n\n"
                )
                idx += 1

        third_llm_prompt = (
            "CURRENT DOCUMENT:\n\n"

            f"{self.document_draft}\n\n"

            "SECTIONS (FROM L2 BOT):\n\n"

            f"{section_output}\n\n"

            "TASK:\n\n"

            r"You are on Step 3: Drafting a copy of the document by inserting the work of the L2 bots." "\n\n"

            "You need to update the current document draft to include the new section. Note that, while "
            "you need to write the preamble, title, abstract, and can add introductory paragraphs, explaining " 
            "what this document will cover, you cannot write any NEW sections here. Most of the mathematics " 
            "will be copy and pasted from the provided sections.\n\n"

            r"You can do so by writing '\section{X}' on a line by itself, where X is the name of the "
            "section that you would like to go there. This will paste the ENTIRE section there once "
            "you are done writing. You do not need to write ANY of the content of the section " 
            "besides its title.\n\n"

            r"You should NEVER write \subsection{Y}, because the subsections are already naturally nested "
            "inside of the sections that you will be pasting."
        )

        self.draft_document_code = await llm_call(
            prompt=third_llm_prompt,
            system_prompt=self.system_prompt
        )

        if Config.L1_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L1 Bot Iteration #{self.iterations + 1} - Draft Document Code\n" +
                "-" * 50 + "\n" +
                self.draft_document_code + "\n" +
                "=" * 50 + "\n"
            )
            print(output)
 
        # Identify the section titles actually used in the draft document
        used_titles = set(re.findall(r'\\section\{([^}]+)\}', self.draft_document_code))

        # Discard sections that are not referenced in the draft document
        self.section_blocks = {title: block for title, block in self.section_blocks.items() if title in used_titles}

        # Replace placeholders with actual section content.
        def replace_section(match_obj):
            title = match_obj.group(1).strip()
            return self.section_blocks.get(title, match_obj.group(0))

        self.document_draft = re.sub(
            r'\\section\{([^}]+)\}', 
            replace_section, 
            self.draft_document_code
        )

        if Config.L1_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L1 Bot Iteration #{self.iterations + 1} - Document Draft\n" +
                "-" * 50 + "\n" +
                self.document_draft + "\n" +
                "=" * 50 + "\n"
            )
            print(output)


    async def _final_decision(self):
        """LLM call to determine if the updated document is complete or needs further refinement."""
        final_llm_prompt = (
            "CURRENT DOCUMENT DRAFT:\n\n"

            f"{self.document_draft}\n\n"

            "TASK:\n\n"

            r"You are on Step 4: Deciding whether the document is in satisfactory " 
            "shape or needs to go through further revisions.\n\n"

            "You have produced the given document above. You have the option to keep editing it "
            "and adding new sections, or mark the document as complete. It should be relatively clear "
            "whether or not the current document can be improved upon, or if it is in satisfactory "
            "shape given its abstract.\n\n"

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

        if Config.L1_PRINT:
            output = (
                "\n" + "=" * 50 + "\n" +
                f"L1 Bot Iteration #{self.iterations + 1} - Final Decision\n" +
                "-" * 50 + "\n" +
                self.final_decision_response + "\n" +
                "=" * 50 + "\n"
            )
            print(output)

        self.iterations += 1

        # Final evaluation to decide if the document is complete.
        final_line = self.final_decision_response.splitlines()[-1].strip().upper()
        if ("COMPLETE" in final_line) or (not self.iterations < Config.L1_REASONING_STEPS):
            self.done = True


    async def step(self):
        """
        Execute the next LLM call in the sequence.
        Each call to this method will result in exactly one llm_call.
        """
        if self.done:
            raise RuntimeError("L1 bot called after being marked done.")
        if self.iterations >= Config.L1_REASONING_STEPS:
            raise RuntimeError("Iteration limit reached: Maximum number of L1 reasoning steps exceeded.")

        # Sequence of LLM calls in one iteration.
        llm_call_sequence = [
            self._preliminary_reasoning,
            self._section_reasoning_list,
            self._format_instructions_for_L2,
        ]

        # If multiple L3 bots are configured, incorporate round robin tournaments.
        if Config.NUM_L2_BOTS > 1:
            llm_call_sequence.extend([
                self._create_round_robin,
            ])

        # Append the remaining steps.
        llm_call_sequence.extend([
            self._draft_document_code,
            self._final_decision,
        ])

        # Execute the current LLM call.
        await llm_call_sequence[self.current_llm_call_index]()

        # Move to the next step in the sequence.
        self.current_llm_call_index += 1
        # If we have completed all 5 calls in this iteration, reset for the next iteration.
        if self.current_llm_call_index >= len(llm_call_sequence):
            self.current_llm_call_index = 0

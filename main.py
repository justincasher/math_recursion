# main.py

import asyncio
import threading
import tkinter as tk

from config import Config
from bots.L1_bot import L1Bot
from bots.L2_bot import L2Bot
from bots.L3_bot import L3Bot
from bots.L4_bot import L4Bot
from latex_labels import LabelManager
from visualizer import Visualizer


def get_leaves(level_bot):
    """
    Recursively retrieves all leaf nodes from a tree of bots that are not marked as done,
    with an added filtering step. For the children of each node, groups are formed based on
    the combination of L2_instruction, L3_instruction, and L4_instruction. If any leaf in a
    group is marked as done, all leaves in that group are marked as done before the recursion
    proceeds.

    A leaf node is defined as a bot that:
      - Is not already marked as done.
      - Either does not have a 'children' attribute or its children are all marked as done.

    Parameters:
        level_bot (object): An instance of a bot in the bot tree. It is expected to have a 'done'
                            attribute (a boolean) and optionally a 'children' attribute (list).

    Returns:
        list: A list of leaf bot nodes that are eligible for further updates.
              Nodes that are already marked as done are skipped.
    """
    # If the current bot is done, it doesn't contribute any leaves.
    if getattr(level_bot, 'done', False):
        return []

    # Retrieve children list if it exists; default to empty list otherwise.
    children = getattr(level_bot, 'children', [])

    # If there are children, first group them by the combined instructions.
    if children:
        groups = {}
        for child in children:
            key = (
                getattr(child, 'L2_instruction', None),
                getattr(child, 'L3_instruction', None),
                getattr(child, 'L4_instruction', None)
            )
            groups.setdefault(key, []).append(child)

        # For each group, if any child is marked as done, mark every child in that group as done.
        for group in groups.values():
            if len(group) > 1 and any(getattr(child, 'done', False) for child in group):
                for child in group:
                    child.done = True

    # Re-fetch children in case any have been marked as done.
    children = getattr(level_bot, 'children', [])

    # Determine if this node is a leaf:
    # It is a leaf if there are no children or all children are marked as done.
    if not children or all(getattr(child, 'done', False) for child in children):
        return [level_bot]

    # Otherwise, recursively collect leaves from children.
    leaves = []
    for child in children:
        leaves.extend(get_leaves(child))

    return leaves


async def parallel_updates(L1, visualizer=None):
    """
    Asynchronously runs the bot update steps in a loop while optionally updating the visualizer.
    
    This function pauses if the visualizer is paused.
    
    Parameters:
        L1: The head bot instance.
        visualizer: The Visualizer instance (or None if not used).
    """
    while L1.iterations < Config.L1_REASONING_STEPS:
        # 1) Retrieve leaves of the bot tree.
        leaves = get_leaves(L1)
        
        # 2) Execute the async step method of each leaf concurrently.
        tasks = [asyncio.create_task(bot.step()) for bot in leaves]
        await asyncio.gather(*tasks)
        
        # 3) Update the visualizer with the latest bot state, if it exists.
        if visualizer is not None:
            visualizer.update()
        
        # 4) Pause the loop if the visualizer is in a paused state.
        if visualizer is not None:
            while visualizer.paused:
                await asyncio.sleep(0.1)


async def sequential_updates(L1, visualizer=None):
    """
    Asynchronously runs the bot update steps in a loop sequentially, one at a time.
    
    Before each bot step, it retrieves the next leaf (using a depth-first approach) from the bot tree.
    This function pauses if the visualizer is paused.
    
    Parameters:
        L1: The head bot instance.
        visualizer: The Visualizer instance (or None if not used).
    """
    while L1.iterations < Config.L1_REASONING_STEPS:
        if visualizer is not None:
            visualizer.update()
        
        if visualizer is not None:
            while visualizer.paused:
                await asyncio.sleep(0.1)
                
        leaves = get_leaves(L1)
        if leaves:
            # Process the next leaf in a depth-first manner.
            next_leaf = leaves[0]
            await next_leaf.step()
        else:
            # If no leaf is available, wait briefly before retrying.
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    # 1. Load the existing LaTeX document and instruction.
    with open(Config.DOC_LOAD, "r", encoding="utf-8") as f:
        original_document = f.read()

    with open(Config.DOC_INSTRUCTION, "r", encoding="utf-8") as f:
        doc_instruction = f.read()

    # 2. Initialize the L1 bot with the document and label manager.
    lbl_mgr = LabelManager(original_document)
    L1 = L1Bot(
        original_document,
        doc_instruction,
        lbl_mgr
    )
    
    if Config.VISUALIZER:
        # 3a. Create the Visualizer instance (this sets up the Tkinter GUI).
        visualizer = Visualizer(L1)
        
        # 4a. Run the async update loop in a background thread.
        def run_async_updates():
            if Config.PARALLEL:
                asyncio.run(parallel_updates(L1, visualizer))
            else:
                asyncio.run(sequential_updates(L1, visualizer))
        
        update_thread = threading.Thread(target=run_async_updates, daemon=True)
        update_thread.start()
        
        # 5a. Run the Tkinter main loop on the main thread.
        visualizer.root.mainloop()
        
        # 6a. Once the GUI window is closed, join the update thread.
        update_thread.join()
    else:
        # 3b. Run updates without the visualizer.
        if Config.PARALLEL:
            asyncio.run(parallel_updates(L1, None))
        else:
            asyncio.run(sequential_updates(L1, None))
    
    # 7. Save the final updated document to math_new.txt.
    updated_document = L1.document_draft
    with open(Config.DOC_SAVE, "w", encoding="utf-8") as f:
        f.write(updated_document)
    
    print("Updated LaTeX document has been saved to math_new.txt.")

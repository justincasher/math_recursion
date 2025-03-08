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
    Recursively retrieves all leaf nodes from a tree of bots that are not marked as done.

    A leaf node is defined as a bot that:
      - Is not already marked as done.
      - Either does not have 'children' or 'round_robin' attributes, or both are empty,
        or all bots in these lists are marked as done.

    Parameters:
        level_bot (object): An instance of a bot in the bot tree. It is expected to have a 'done' 
                            attribute (a boolean) and optionally 'children' and 'round_robin' attributes (lists).

    Returns:
        list: A list of leaf bot nodes that are eligible for further updates.
              Nodes that are already marked as done are skipped.
    """
    if getattr(level_bot, 'done', False):
        return []
    
    # Retrieve children and round_robin lists if they exist; default to empty lists otherwise.
    children = getattr(level_bot, 'children', [])
    round_robin = getattr(level_bot, 'round_robin', [])
    
    # Determine if this node is a leaf:
    # It is a leaf if both children and round_robin are empty,
    # or if all bots in both lists are marked as done.
    if (not children and not round_robin) or \
       (all(getattr(child, 'done', False) for child in children) and 
        all(getattr(child, 'done', False) for child in round_robin)):
        return [level_bot]
    
    leaves = []
    for child in children:
        leaves.extend(get_leaves(child))
    for child in round_robin:
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
            # Process the next leaf in a depth-first manner
            next_leaf = leaves[0]
            await next_leaf.step()
        else:
            # If no leaf is available, wait briefly before retrying
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    # 1. Load the existing LaTeX document from math.txt
    with open(Config.DOC_LOAD, "r", encoding="utf-8") as f:
        original_document = f.read()

    # 2. Initialize the L1 bot with the document and label manager.
    lbl_mgr = LabelManager(original_document)
    L1 = L1Bot(original_document, lbl_mgr)
    
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

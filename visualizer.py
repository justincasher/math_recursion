import tkinter as tk
from config import Config

class Visualizer:
    def __init__(self, l1_bot):
        """
        Initialize the Visualizer with the L1 bot instance.
        Sets up a two-column window:
          - Left side: displays only the actual math/document content.
                      This panel has a fixed width determined by Config.MAX_DISPLAY.
          - Right side: displays meta information (iterations), instruction (in a code environment),
                        and navigation buttons that dynamically resize.
        """
        self.l1_bot = l1_bot
        self.paused = False
        self.history = []  # For back navigation support
        self.current_view = (l1_bot, "L1")  # Initial view is the document

        self.root = tk.Tk()
        self.root.title("Bot Visualizer")
        self.root.geometry("1200x800")

        # Main container frame holding left and right columns.
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        # Left frame: displays the document/math content.
        # Its width is fixed to Config.MAX_DISPLAY pixels.
        self.left_frame = tk.Frame(self.main_frame, width=Config.MAX_DISPLAY)
        self.left_frame.pack(side=tk.LEFT, fill="both", padx=10, pady=10)
        self.left_frame.pack_propagate(False)  # Prevent resizing based on contents

        # Create a vertical scrollbar and attach it to the Text widget.
        self.scrollbar = tk.Scrollbar(self.left_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

        self.details_text = tk.Text(self.left_frame, wrap="word", yscrollcommand=self.scrollbar.set)
        self.details_text.pack(side=tk.LEFT, fill="both", expand=True)
        self.scrollbar.config(command=self.details_text.yview)

        # Right frame: displays meta information, instruction (as code), and navigation controls.
        # It expands dynamically when the window is resized.
        self.nav_frame = tk.Frame(self.main_frame)
        self.nav_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=10, pady=10)
        # Bind a configure event to adjust button wraplengths dynamically.
        self.nav_frame.bind("<Configure>", self.adjust_nav_widgets)

        # Pause/Resume Updates button below the main frame.
        self.pause_button = tk.Button(self.root, text="Pause Updates", command=self.toggle_pause)
        self.pause_button.pack(pady=5)

        # Initialize the view with the L1 document.
        self.update_view(l1_bot, "L1")

    def adjust_nav_widgets(self, event):
        """
        Adjust the wraplength for all button widgets inside the navigation frame
        based on the current width of the nav_frame.
        """
        new_width = event.width - 20  # Subtract a margin for padding
        for widget in self.nav_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.configure(wraplength=new_width)

    def update_wraplengths(self):
        """
        Force update the wraplength for buttons using the current width of nav_frame.
        This is used when the window is first displayed and no <Configure> event occurs.
        """
        # Ensure that geometry information is up-to-date
        self.nav_frame.update_idletasks()
        new_width = self.nav_frame.winfo_width() - 20
        for widget in self.nav_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.configure(wraplength=new_width)

    def toggle_pause(self):
        """Toggle the pause/resume state of updates."""
        self.paused = not self.paused
        self.pause_button.config(text="Resume Updates" if self.paused else "Pause Updates")

    def update_view(self, bot, bot_type):
        """
        Update the left (content) and right (meta/navigation) panels based on the current bot view.
        
        The left panel displays only the raw math/document content (without extra labels),
        while the right panel shows meta data such as iterations and the instruction (in a code environment),
        as well as navigation buttons.
        """
        # Update left panel with only the math/document content.
        self.details_text.config(state="normal")
        self.details_text.delete("1.0", tk.END)
        if bot_type == "L1":
            left_content = getattr(bot, 'document_draft', '')
        elif bot_type == "L2":
            left_content = getattr(bot, 'section_draft', '')
        elif bot_type == "L3":
            left_content = getattr(bot, 'subsection_draft', '')
        elif bot_type == "L4":
            left_content = getattr(bot, 'math_draft', '')
        elif bot_type == "Review":
            left_content = getattr(bot, 'summary', 'No summary available.')
        else:
            left_content = ''
        self.details_text.insert(tk.END, left_content)
        self.details_text.config(state="disabled")

        # Clear the right panel.
        for widget in self.nav_frame.winfo_children():
            widget.destroy()

        # Display meta information.
        meta_info = f"Iterations: {getattr(bot, 'iterations', 'N/A')}"
        meta_label = tk.Label(self.nav_frame, text=meta_info)
        meta_label.pack(pady=5, fill="x")

        # Get and display the instruction.
        if bot_type in ("L1", "Review"):
            instruction_text = getattr(bot, 'instruction', '')
        elif bot_type == "L2":
            instruction_text = getattr(bot, 'L2_instruction', '')
        elif bot_type == "L3":
            instruction_text = getattr(bot, 'L3_instruction', '')
        elif bot_type == "L4":
            instruction_text = getattr(bot, 'L4_instruction', '')
        else:
            instruction_text = ''

        if instruction_text:
            instruction_label = tk.Label(self.nav_frame, text="instruction:")
            instruction_label.pack(pady=(10, 0), fill="x")
            instruction_frame = tk.Frame(self.nav_frame, height=Config.INSTRUCTION_HEIGHT)
            instruction_frame.pack(pady=5, fill="x")
            instruction_frame.pack_propagate(False)
            code_widget = tk.Text(instruction_frame, wrap="word", font=("Courier", 14))
            code_widget.insert(tk.END, instruction_text)
            code_widget.configure(state="disabled")
            code_widget.pack(fill="both", expand=True)

        # Back button if available.
        if self.history:
            back_btn = tk.Button(self.nav_frame, text="Back", command=self.go_back,
                                 justify="left", anchor="w")
            back_btn.pack(pady=5, fill="x")

        # Navigation buttons for child bots.
        if hasattr(bot, 'children'):
            for i, child in enumerate(bot.children):
                if bot_type == "L1":
                    child_instruction = getattr(child, 'L2_instruction', 'No instruction')
                    btn_label = f"Section {i + 1}: {child_instruction.splitlines()[0] if child_instruction else 'No Title'}"
                    child_type = "L2"
                elif bot_type == "L2":
                    child_instruction = getattr(child, 'L3_instruction', 'No instruction')
                    btn_label = f"Subsection {i + 1}: {child_instruction.splitlines()[0] if child_instruction else 'No Title'}"
                    child_type = "L3"
                elif bot_type == "L3":
                    child_instruction = getattr(child, 'L4_instruction', 'No instruction')
                    btn_label = f"Block {i + 1}: {child_instruction.splitlines()[0] if child_instruction else 'No Title'}"
                    child_type = "L4"
                elif bot_type == "L4":
                    child_instruction = getattr(child, 'instruction', 'No instruction')
                    btn_label = f"Review Bot {i + 1}: {child_instruction.splitlines()[0] if child_instruction else 'No Title'}"
                    child_type = "Review"
                else:
                    child_type = None

                if child_type:
                    btn = tk.Button(
                        self.nav_frame,
                        text=btn_label,
                        justify="left",   # Allow multiline text alignment
                        anchor="w",       # Anchor text to the left
                        command=lambda c=child, ct=child_type: self.navigate_to(c, ct)
                    )
                    btn.pack(pady=2, fill="x")

        # Save the current view.
        self.current_view = (bot, bot_type)

        # Force an update of wraplengths once the nav_frame is drawn.
        self.update_wraplengths()

    def navigate_to(self, bot, bot_type):
        """
        Navigate to a child bot view.
        Push the current view onto the history stack, then update the display.
        """
        self.history.append(self.current_view)
        self.update_view(bot, bot_type)

    def go_back(self):
        """
        Navigate back to the previous view.
        Pops the last view from the history stack and updates the display.
        """
        if self.history:
            previous_view = self.history.pop()
            self.update_view(previous_view[0], previous_view[1])

    def refresh(self):
        """
        Refresh the current view with up-to-date bot information.
        Only updates if not paused.
        """
        if self.paused:
            return
        bot, bot_type = self.current_view
        self.update_view(bot, bot_type)

    def update(self):
        """
        Public update method to be called by the main loop after each processing round.
        A delay is always scheduled so that the UI remains responsive even when updates are paused.
        """
        self.root.after(100, self.refresh)

import tkinter as tk
from tkinter import ttk, messagebox


class SymbioticSystemUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Symbiotic System")
        self.root.geometry("700x500")
        self.root.minsize(650, 450)

        # Variable to store the Keycloak ID
        self.user_keycloak_id = None

        self.setup_styles()
        self.build_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Title.TLabel",
            font=("Arial", 20, "bold"),
            padding=10
        )

        style.configure(
            "Header.TLabel",
            font=("Arial", 12, "bold"),
            padding=5
        )

        style.configure(
            "Body.TLabel",
            font=("Arial", 11),
            padding=3
        )

        style.configure(
            "Custom.TButton",
            font=("Arial", 11, "bold"),
            padding=8
        )

        style.configure(
            "Custom.TEntry",
            padding=6
        )

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Welcome to the Symbiotic System",
            style="Title.TLabel",
            anchor="center"
        )
        title_label.pack(pady=(0, 15))

        # Welcome note
        welcome_text = (
            "This interface will help you begin the experiment setup.\n"
            "Please create your Keycloak ID first, then enter it below."
        )
        welcome_label = ttk.Label(
            main_frame,
            text=welcome_text,
            style="Body.TLabel",
            justify="center"
        )
        welcome_label.pack(pady=(0, 20))

        # Instructions section
        instructions_frame = ttk.LabelFrame(main_frame, text="Instruction Guide", padding=15)
        instructions_frame.pack(fill="x", pady=10)

        instructions = [
            "1. Open the Keycloak registration page.",
            "2. Create your user account using your assigned credentials.",
            "3. After account creation, note down your Keycloak ID.",
            "4. Return to this interface and enter your Keycloak ID below.",
            "5. Click 'Save ID' to store it for the experiment."
        ]

        for instruction in instructions:
            ttk.Label(
                instructions_frame,
                text=instruction,
                style="Body.TLabel",
                justify="left"
            ).pack(anchor="w")

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="User ID Entry", padding=15)
        input_frame.pack(fill="x", pady=20)

        ttk.Label(
            input_frame,
            text="Enter your Keycloak ID:",
            style="Header.TLabel"
        ).pack(anchor="w", pady=(0, 8))

        self.keycloak_entry = ttk.Entry(input_frame, width=40, style="Custom.TEntry")
        self.keycloak_entry.pack(anchor="w", pady=(0, 12))
        self.keycloak_entry.focus()

        button_frame = ttk.Frame(input_frame)
        button_frame.pack(anchor="w", pady=5)

        save_button = ttk.Button(
            button_frame,
            text="Save ID",
            style="Custom.TButton",
            command=self.save_keycloak_id
        )
        save_button.pack(side="left", padx=(0, 10))

        show_button = ttk.Button(
            button_frame,
            text="Show Stored ID",
            style="Custom.TButton",
            command=self.show_stored_id
        )
        show_button.pack(side="left")

        # Output section
        self.output_label = ttk.Label(
            main_frame,
            text="No Keycloak ID stored yet.",
            style="Header.TLabel",
            foreground="blue"
        )
        self.output_label.pack(pady=20)

    def save_keycloak_id(self):
        entered_id = self.keycloak_entry.get().strip()

        if not entered_id:
            messagebox.showwarning("Input Error", "Please enter a valid Keycloak ID.")
            return

        self.user_keycloak_id = entered_id
        self.output_label.config(
            text=f"Stored Keycloak ID: {self.user_keycloak_id}"
        )

        messagebox.showinfo(
            "Success",
            f"Keycloak ID '{self.user_keycloak_id}' has been stored successfully."
        )

    def show_stored_id(self):
        if self.user_keycloak_id is None:
            messagebox.showinfo("Stored ID", "No Keycloak ID has been stored yet.")
        else:
            messagebox.showinfo(
                "Stored ID",
                f"Current stored Keycloak ID: {self.user_keycloak_id}"
            )


def main():
    root = tk.Tk()
    app = SymbioticSystemUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
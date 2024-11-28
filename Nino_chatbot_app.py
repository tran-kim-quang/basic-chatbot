from tkinter import *
from tkinter import font
from Nino import get_chat, name

BG_GRAY = "#ff8fab"
BG_COLOR = "#ffb3c6"
TEXT_COLOR = "#000000"

class Nino_app:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        
        FONT = font.Font(family="Times New Roman", size=14)
        FONT_BOLD = font.Font(family="Times New Roman", size=13, weight="bold")

        self.window.title("Nino")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=1920, height=1080, bg=BG_COLOR)

        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Ohaio :3", font=font.Font(family="Times New Roman", size=20, weight="bold"), pady=20)
        head_label.place(relwidth=1)

        line=Label(self.window, width=1900, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.08, relheight=0.012)

        #text
        self.text_widget = Text(self.window, width=25, height=4, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.73, relwidth=1, rely=0.09)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        #scroll
        scroll = Scrollbar(self.text_widget)
        scroll.place(relheight=1, relx=0.974)
        scroll.configure(command=self.text_widget.yview)

        #bot label
        bot_label = Label(self.window, bg=BG_GRAY, height=80)
        bot_label.place(relwidth=1, rely=0.825)

        #mess box
        self.msg_entry = Entry(bot_label, bg="#ffffff", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.725, relheight=0.06, rely=0.05, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_press)

        #button
        send_button = Button(bot_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_press(None))
        send_button.place(relx=0.77, rely=0.03, relwidth=0.2, relheight=0.09)

    def _on_enter_press(self, event):
        msg = self.msg_entry.get()
        self._insert_msg(msg, "You")
        return
    
    def _insert_msg(self, msg, send):
        if not msg:
            return 

        self.msg_entry.delete(0, END)
        msg1 = f"{send}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"{name}: {get_chat(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

if __name__ == "__main__":
    app = Nino_app()
    app.run()
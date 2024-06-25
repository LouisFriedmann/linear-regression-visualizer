# button.py allows the user to click buttons

from matplotlib.widgets import Button

class InteractiveButton:
    def __init__(self, fig, ax, color, hovercolor, text, submit_func, enabled):
        self.fig = fig
        self.ax = ax
        self.button = Button(self.ax, str(text), color=color, hovercolor=hovercolor)

        self.submit_func = submit_func
        self.conn_id = self.button.on_clicked(self.submit_func)
        self.enabled = enabled

    def disconnect(self):
        self.button.disconnect(self.conn_id)
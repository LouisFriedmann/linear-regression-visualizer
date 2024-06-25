# handle_data_entry.py allows user to enter data points user sliders

from matplotlib.widgets import Slider
from button import InteractiveButton
from text import FigText

def handle_data_entry(fig, ax, plt, min_points, max_points):
    data_entry_obj = HandleDataEntry(fig, ax, min_points)
    while not data_entry_obj.is_animation_button_clicked and data_entry_obj.get_points_added() < max_points:
        plt.pause(.1)

    data_entry_obj.limit_points_text.remove()
    data_entry_obj.disable_buttons_and_sliders()
    plt.pause(1)
    return data_entry_obj.get_x_and_y_coords()

class HandleDataEntry:
    def __init__(self, fig, ax, min_points):
        self.fig = fig
        self.ax = ax
        self.min_points = min_points

        self.points_added = 0
        self.x_coords = []
        self.y_coords = []

        self.x_slider = Slider(self.fig.add_axes((.35, .05, .3, .1)), "X coordinate:", valmin=-100, valmax=100, valinit=50,
                          valstep=1)
        self.y_slider = Slider(self.fig.add_axes((.35, -.02, .3, .1)), "Y coordinate:", valmin=-100, valmax=100, valinit=50,
                          valstep=1)

        self.add_button = InteractiveButton(fig=fig, ax=fig.add_axes((.7, .015, .13, .1)), color="green", hovercolor="lightgreen", text="Add",
                                            submit_func=self.add_button_submit, enabled=True)

        self.limit_points_text = FigText(x=.5, y=.5, txt="Add between 3 and 6 data points", fig=fig, fontsize=20)
        self.is_animation_button_clicked = False

    def update_x(self, val):
        print(f"X slider value: {self.x_slider.val}")

    def update_y(self, val):
        print(f"Y slider value: {self.y_slider.val}")

    def add_button_submit(self, event):
        x = self.x_slider.val
        y = self.y_slider.val

        # Check to make sure the point is unique
        if not x in self.x_coords or not y in self.y_coords:
            # Place x value such that x_coords remains sorted and pair y with x
            if not self.x_coords:
                self.x_coords.append(x)
                self.y_coords.append(y)
            else:
                for i, x_coord in enumerate(self.x_coords):
                    if x <= x_coord:
                        self.x_coords.insert(i, x)
                        self.y_coords.insert(i, y)
                        break

                    elif i == len(self.x_coords) - 1:
                        self.x_coords.append(x)
                        self.y_coords.append(y)
                        break

            self.ax.plot([x], [y], "ro")
            self.points_added += 1

        # User can play animation when they've entered 'self.min_points' or more data points: button will pop up when this happens
        if self.points_added == self.min_points:
            self.play_animation_button = InteractiveButton(fig=self.fig, ax=self.fig.add_axes((.85, .015, .13, .1)), color="blue",
                                                           hovercolor="lightblue", text="Play Animation",
                                                           submit_func=self.play_animation_button_submit, enabled=True)

    def play_animation_button_submit(self, event):
        self.is_animation_button_clicked = True

    def disable_buttons_and_sliders(self):
        self.add_button.disconnect()
        self.play_animation_button.disconnect()
        self.x_slider.set_active(False)
        self.y_slider.set_active(False)

    def get_points_added(self):
        return self.points_added

    def get_x_and_y_coords(self):
        return self.x_coords, self.y_coords
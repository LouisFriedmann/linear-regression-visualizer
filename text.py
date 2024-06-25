# text.py contains the class for manipulating matplotlib text

import matplotlib.pyplot as plt

class FigText:
    def __init__(self, x, y, txt, fig, wrap=False, fontsize=10, rotate=False):
        self.x, self.y = x, y
        self.txt_str = txt
        self.fig = fig
        self.fontsize = fontsize
        if rotate:
            rotation_mode = "anchor"
        else:
            rotation_mode = "default"

        self.txt = fig.text(self.x, self.y, self.txt_str, transform=self.fig.transFigure, rotation=rotate, wrap=wrap,
                            rotation_mode=rotation_mode, fontsize=self.fontsize)


    # move text to a location on the axes in a specified amount of time
    def move(self, x, y, seconds=0, total_frames=1):
        crnt_x, crnt_y = self.x, self.y
        dest_x, dest_y = x, y
        dx, dy = (dest_x - crnt_x) / total_frames, (dest_y - crnt_y) / total_frames
        delay = seconds / total_frames
        for i in range(total_frames):
            if delay != 0:
                plt.pause(delay)
            self.x += dx
            self.y += dy
            self.txt.set_position((self.x, self.y))

    def rotate(self, seconds=0, total_frames=1):
        delay = seconds / total_frames
        next_rotation_amt = 360 / total_frames
        rot_amount = next_rotation_amt
        for i in range(total_frames):
            if delay != 0:
                plt.pause(delay)

            self.txt.set_rotation(rot_amount)
            rot_amount += next_rotation_amt

    def set_text(self, txt):
        self.txt_str = txt
        self.txt.set_text(self.txt_str)

    def set_fontsize(self, size):
        self.fontsize = size
        self.txt.set_fontsize(self.fontsize)

    def remove(self):
        self.txt.remove()

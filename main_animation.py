# main_animation.py handles the logic for the main animation after using enters their data points

import numpy as np

from text import FigText
from matplotlib.animation import FuncAnimation

TIMES_TO_DRAW = 20
VERTICAL_TEXT_SPACE = .1

class MainAnimation:
    def __init__(self, plt, ax, ax2, fig, x_coords, y_coords):
        self.plt = plt
        self.ax = ax
        self.ax2 = ax2
        self.fig = fig
        self.current_step = 0
        self.steps = [
            self.handle_subscripts_and_polynomial,
            self.handle_plot_polynomial,
            self.handle_plot_residuals,
            self.zoom_middle_residual,
            self.move_ith_residual_text,
            self.sum_minimized,
            self.show_derivatives,
            self.derivative_animation,
            self.set_up_system_of_equations,
            self.show_solved_A_and_B,
            self.evaluate_sigma_expressions,
            self.animate_to_best_fit,
            self.thanks_for_watching
        ]
        
        self.x_coords, self.y_coords = x_coords, y_coords
        
        self.degree = 1 # since this is LINEAR regression

        self.x_polynomial, self.y_polynomial = [], []

    def handle_subscripts_and_polynomial(self):
        self.x_coords_array, self.y_coords_array = np.array(self.x_coords), np.array(self.y_coords)

        # Display the subscripts of the first, middle, and last point coordinates
        self.x1_y1_text = self.plt.text(self.x_coords[0], self.y_coords[0], r'$(x_1, y_1)$', transform=self.ax.transData)
        self.fig.canvas.draw()
        self.plt.pause(1)
        self.plt.text(self.x_coords[int(len(self.x_coords) / 2)], self.y_coords[int(len(self.x_coords) / 2)],
                    r'$(x_i, y_i)$', transform=self.ax.transData)
        self.fig.canvas.draw()
        self.plt.pause(1)
        self.xn_yn_text = self.plt.text(self.x_coords[-1], self.y_coords[-1], r'$(x_n, y_n)$', transform=self.ax.transData)
        self.fig.canvas.draw()
        self.plt.pause(1)

        self.__display_polynomial()

    def __display_polynomial(self):
        # Display the linear polynomial at the top
        self.y_fit_coefficients = np.polyfit(self.x_coords_array, self.y_coords_array, self.degree)

        self.x_continuous = np.linspace(self.x_coords_array[0], self.x_coords_array[-1], num=TIMES_TO_DRAW)

        self.y_fit = np.polyval(self.y_fit_coefficients, self.x_continuous)
        self.to_become_y_fit = 2 * self.y_fit

        self.general_predict_equation = "ŷ = Ax + B"
        rounded_scaled_coefficients = [round(coef, 2) * 2 for coef in list(self.y_fit_coefficients)]
        self.fit_eq_str = f"ŷ = {rounded_scaled_coefficients[0]}x + {rounded_scaled_coefficients[1]}"  # Will eventually be the string version of the equation of best fit line

        self.fit_eq_text = FigText(x=self.ax.get_position().x0, y=self.ax.get_position().y0 + self.ax.get_position().height,
                              txt=self.fit_eq_str, fig=self.fig, fontsize=13)

        self.general_predict_equation_text = FigText(x=self.ax.get_position().x0,
                                                y=self.ax.get_position().y0 + .05 + self.ax.get_position().height,
                                                txt=self.general_predict_equation, fig=self.fig, fontsize=13)

    def handle_plot_polynomial(self):
        self.lines = [] # Store references for all the lines plotted for the polynomial to be removed later on

        # Plot the polynomial that will eventually fit the data points
        for i in range(TIMES_TO_DRAW):
            self.x_polynomial.append(self.x_continuous[i])
            self.y_polynomial.append(self.to_become_y_fit[i])
            line = self.ax.plot(self.x_polynomial, self.y_polynomial, color='g', alpha=0.5)[0]
            self.lines.append(line)
            self.plt.pause(.1)

    def handle_plot_residuals(self):
        x_to_plot = []
        y_to_plot = []
        seconds_plot_residuals = 3
        self.to_become_y_fit_coefficients = self.y_fit_coefficients * 2
        self.residual_line_info = [[(x, y), (x, np.polyval(self.to_become_y_fit_coefficients, x))] for x, y in zip(self.x_coords, self.y_coords)]
        seconds_per_plot = seconds_plot_residuals / len(self.residual_line_info)

        self.residual_lines = [] # Store references for all the residual lines plotted here to be removed later

        for i, line in enumerate(self.residual_line_info):
            x = line[0][0]
            y = line[0][1]
            y_predict = line[1][1]
            y_continuous = np.linspace(y_predict, y, num=TIMES_TO_DRAW)
            for y in y_continuous:
                x_to_plot.append(x)
                y_to_plot.append(y)
            line = self.ax.plot(x_to_plot, y_to_plot, color='r', alpha=0.5)[0]
            self.residual_lines.append(line)
            x_to_plot.clear()
            y_to_plot.clear()
            self.plt.pause(seconds_per_plot)

    def zoom_middle_residual(self):
        # First hide the coordinates (X1, Y1) and (Xn, Yn)
        self.x1_y1_text.set_text("")
        self.xn_yn_text.set_text("")

        # zoom in on the middle residual
        times_to_zoom = 20
        seconds_to_zoom = 3
        delay = seconds_to_zoom / times_to_zoom
        y_data_point = self.y_coords[int(len(self.x_coords) / 2)]
        y_mid_predict = self.residual_line_info[int(len(self.x_coords) / 2)][1][1]
        if y_data_point < y_mid_predict:
            zoomed_y_start = y_data_point
            zoomed_y_end = y_mid_predict
        else:
            zoomed_y_start = y_mid_predict
            zoomed_y_end = y_data_point

        self.original_x_lims = self.ax.get_xlim() # To use when zooming back out
        self.original_y_lims = self.ax.get_ylim() # To use when zooming back out

        mid_x = self.x_coords[int(len(self.x_coords) / 2)]
        zoomed_x_start, zoomed_x_end = mid_x - (zoomed_y_end - zoomed_y_start) / 2, mid_x + (zoomed_y_end - zoomed_y_start) / 2
        x_start_step, x_end_step = (zoomed_x_start - self.ax.get_xlim()[0]) / times_to_zoom, (zoomed_x_end - self.ax.get_xlim()[1]) / times_to_zoom
        y_start_step, y_end_step = (zoomed_y_start - self.ax.get_ylim()[0]) / times_to_zoom, (zoomed_y_end - self.ax.get_ylim()[1]) / times_to_zoom

        # circle the middle residual before zooming
        radius = (zoomed_y_end - zoomed_y_start) / 2
        self.circle = self.plt.Circle((mid_x, zoomed_y_start + radius), radius, color='b', fill=False)
        self.ax.add_patch(self.circle)

        # Zoom in on the middle residual
        self.x_start, self.x_end = self.ax.get_xlim()
        self.y_start, self.y_end = self.ax.get_ylim()
        for _ in range(times_to_zoom):
            self.x_start += x_start_step
            self.x_end += x_end_step
            self.y_start += y_start_step
            self.y_end += y_end_step
            self.plt.pause(delay)
            self.ax.set_xlim(self.x_start, self.x_end)
            self.ax.set_ylim(self.y_start, self.y_end)

    def move_ith_residual_text(self):
        self.plt.pause(.5)

        # Moving text for showing the equation of the ith residual
        self.plt.text(self.x_start + (self.x_end - self.x_start) / 2, self.y_end - .2, r'$(x_i, ŷ)$', transform=self.ax.transData)
        self.ax_position = self.ax.get_position()
        self.ax2_position = self.ax2.get_position()
        self.residual_text = FigText(x=self.ax_position.x0 + self.ax_position.width / 2, y=self.ax2_position.y0 + self.ax2_position.height / 2, txt=r'$Residual = |y_i - ŷ|$', fig=self.fig)
        self.plt.pause(1)
        self.residual_text.move(x=self.ax2_position.x0, y=self.ax2_position.y0 + self.ax2_position.height - VERTICAL_TEXT_SPACE, seconds=2, total_frames=20)
        self.residual_text.set_fontsize(20)

        self.sub_y_text = FigText(x=self.ax2_position.x0, y=self.ax2_position.y0, txt="Substitute ŷ", fig=self.fig)
        self.sub_y_text.move(x=self.ax2_position.x0, y=self.ax2_position.y0 + self.ax2_position.height - VERTICAL_TEXT_SPACE * 2,
                        seconds=2, total_frames=20)
        self.sub_y_text.set_fontsize(20)
        self.plt.pause(.5)
        self.residual_substitution_text = FigText(x=self.ax2_position.x0,
                                             y=self.ax2_position.y0 + self.ax2_position.height - VERTICAL_TEXT_SPACE * 4,
                                             txt="Residual = " + "|" + r'$y_i$' + f" - (A" + r'$x_i$' + " + " + "B)|", fig=self.fig, fontsize=20)

    def sum_minimized(self):
        self.ax2_position = self.ax2.get_position()

        # Show the user the equation for the sum of the residuals and tell them the sum must be minimized
        self.plt.pause(2)
        self.residual_text.remove()
        self.sub_y_text.remove()
        self.plt.pause(1)
        self.minimize_sum_text = FigText(x=self.ax2_position.x0 + self.ax2_position.width, y=self.residual_text.y,
                                         txt="Minimize sum(residuals^2)", fig=self.fig, fontsize=20, rotate=True)
        self.minimize_sum_text.move(x=self.residual_text.x, y=self.residual_text.y, seconds=2, total_frames=20)
        self.minimize_sum_text.rotate(seconds=5, total_frames=50)

        self.sigma_text = FigText(x=self.minimize_sum_text.x, y=self.minimize_sum_text.y - VERTICAL_TEXT_SPACE * 2,
                                  txt=r'$\sum_{i=1}^n$' + "(" + r'$y_i$' + " - " + f"(A" + r'$x_i$' + " + " + "B))" + r'$^2$',
                                  fig=self.fig, fontsize=20)
        self.plt.pause(1)
        self.residual_substitution_text.remove()

        # Show taking the derivative with respect to each variable
        self.minimize_sum_se_text = FigText(x=self.minimize_sum_text.x, y=self.minimize_sum_text.y - VERTICAL_TEXT_SPACE * 6,
                                       txt="Take the partial derivative with "
                                           + "respect to each coefficient of the linear polynomial and set them all equal to zero to find "
                                             + "the values of the coefficients such that the entire linear polynomial is minimized", fig=self.fig,
                                            fontsize=20, wrap=True)
        self.plt.pause(7)

    def show_derivatives(self):
        # First delete minimize sum and minimize se sum text and move the sigma expression to previous location of minimize sum text
        minimize_sum_y = self.minimize_sum_text.y
        self.minimize_sum_text.remove()
        self.minimize_sum_se_text.remove()
        self.sigma_text.move(x=self.sigma_text.x, y=minimize_sum_y, seconds=3, total_frames=20)

        # Distribute minus sign for sigma text, store in variable, show that it is an equivalent expression, and update
        # 'sigma_text' to it
        self.distribute_minus_sigma_text = FigText(x=self.sigma_text.x, y=self.sigma_text.y,
                                                   txt="= " + r'$\sum_{i=1}^n$' + "(" + r'$y_i$' + " - " + "A" + r'$x_i $' + " - B)" + r'$^2$', fig=self.fig, fontsize=self.sigma_text.fontsize)
        self.distribute_minus_sigma_text.move(x=self.distribute_minus_sigma_text.x, y=self.distribute_minus_sigma_text.y
                                              - VERTICAL_TEXT_SPACE * 1.5, seconds=3, total_frames=20)
        self.plt.pause(1)
        self.sigma_text.set_text(self.distribute_minus_sigma_text.txt_str.removeprefix("= "))
        self.plt.pause(1)
        self.distribute_minus_sigma_text.remove()


        # Move "A" variable from polynomial at top of axis to the left of sigma text where it'll become "d/dA" of sigma text
        distance_y_predict_to_a = .03
        a_text = FigText(x=self.general_predict_equation_text.x + distance_y_predict_to_a,
                         y=self.general_predict_equation_text.y, txt="A", fig=self.fig)
        a_text.move(x=self.sigma_text.x, y=self.sigma_text.y, seconds=3, total_frames=50)
        a_text.remove()
        distance_move_d_da = .04
        self.d_da_text = FigText(x=self.sigma_text.x, y=self.sigma_text.y, txt=r'$\frac{d}{dA}$', fig=self.fig, fontsize=30)
        self.d_da_text.move(self.d_da_text.x - distance_move_d_da, y=self.sigma_text.y)
        self.plt.pause(.1)
        self.sigma_text.set_text("[" + self.sigma_text.txt_str + "] = 0")

        # Make a clone of sigma text and move it down and do the same thing with "B" as with "A", first distributing minus sign like with 'sigma_text'
        self.sigma_text_clone = FigText(x=self.sigma_text.x, y=self.sigma_text.y,
                                        txt=r'$\sum_{i=1}^n$' + "(" + r'$y_i$' + " - " + f"(A" + r'$x_i$' + " + " + "B))" + r'$^2$',
                                        fig=self.fig, fontsize=self.sigma_text.fontsize)
        self.sigma_text_clone.move(x=self.sigma_text_clone.x, y=self.sigma_text_clone.y - VERTICAL_TEXT_SPACE * 4, seconds=2,
                                   total_frames=20)

        # Distribute minus sign for sigma text clone, store in variable, show that it is an equivalent expression, and update
        # 'sigma_text_clone' to it
        self.distribute_minus_sigma_text_clone = FigText(x=self.sigma_text_clone.x, y=self.sigma_text_clone.y,
                                                   txt="= " + r'$\sum_{i=1}^n$' + "(" + r'$y_i$' + " - " + "A" + r'$x_i$' +  " - B)" + r'$^2$',
                                                   fig=self.fig, fontsize=self.sigma_text_clone.fontsize)
        self.distribute_minus_sigma_text_clone.move(x=self.distribute_minus_sigma_text_clone.x, y=self.distribute_minus_sigma_text_clone.y
                                                                                      - VERTICAL_TEXT_SPACE * 1.5,
                                              seconds=3, total_frames=20)
        self.plt.pause(1)
        self.sigma_text_clone.set_text(self.distribute_minus_sigma_text_clone.txt_str.removeprefix("= "))
        self.plt.pause(1)
        self.distribute_minus_sigma_text_clone.remove()

        distance_y_predict_to_b = .05
        b_text = FigText(x=self.general_predict_equation_text.x + distance_y_predict_to_b,
                         y=self.general_predict_equation_text.y, txt="B", fig=self.fig)
        b_text.move(x=self.sigma_text_clone.x, y=self.sigma_text_clone.y, seconds=3, total_frames=50)
        b_text.remove()
        distance_move_d_db = .04
        self.d_db_text = FigText(x=self.sigma_text_clone.x, y=self.sigma_text_clone.y, txt=r'$\frac{d}{dB}$', fig=self.fig, fontsize=30)
        self.d_db_text.move(self.d_db_text.x - distance_move_d_db, y=self.sigma_text_clone.y)
        self.plt.pause(.1)
        self.sigma_text_clone.set_text("[" + self.sigma_text_clone.txt_str + "] = 0")

    def derivative_animation(self):
        # Show the power rule and chain rule for derivatives
        self.plt.pause(.5)
        self.sigma_derivative_text = FigText(x=self.d_da_text.x, y=self.sigma_text.y - VERTICAL_TEXT_SPACE * 1.5,
                                             txt=r'$\sum_{i=1}^n$' + "2" + "(" + r'$y_i$' + " - A" + r'$x_i$' + " - B)(-" + r'$x_i$' + ")"
                                                  + " = 0",
                                             fig=self.fig, fontsize=self.sigma_text.fontsize)

        self.plt.pause(.5)
        self.sigma_clone_derivative_text = FigText(x=self.d_db_text.x, y=self.sigma_text_clone.y - VERTICAL_TEXT_SPACE * 3,
                                                   txt=r'$\sum_{i=1}^n$' + "2" + "(" + r'$y_i$' + " - A" + r'$x_i $' + " - B)(-1)"
                                                       + " = 0", fig=self.fig, fontsize=self.sigma_text_clone.fontsize)

    def set_up_system_of_equations(self):
        sigma_y = self.sigma_text.y

        # Remove the original sigma expressions with derivatives on them
        self.sigma_text.remove()
        self.sigma_text_clone.remove()
        self.d_da_text.remove()
        self.d_db_text.remove()

        # Align systems of equations in middle of screen
        self.sigma_clone_derivative_text.move(x=self.sigma_clone_derivative_text.x, y=self.sigma_derivative_text.y - VERTICAL_TEXT_SPACE * 1.5,
                                              seconds=3, total_frames=20)
        self.sigma_derivative_text.move(x=self.sigma_derivative_text.x, y=sigma_y, seconds=3, total_frames=20)

        # Show sigma expressions after values from chain rule for differentiation are distributed
        a_text = "-2" + r'$\sum_{i=1}^n$' + r'$x_i$' + r'$y_i$' + " + 2" + r'$\sum_{i=1}^n$' + "B" + r'$x_i$' "+ 2" + r'$\sum_{i=1}^n$' + "A" + r'$x_i^2$' + " = 0"
        b_text = "-2" + r'$\sum_{i=1}^n$' + r'$y_i$' + " + 2" + r'$\sum_{i=1}^n$' + "B + 2" + r'$\sum_{i=1}^n$' + "A" + r'$x_i$' + " = 0"
        self.distribute_sigma_text = FigText(x=self.sigma_derivative_text.x,
                                                   y=self.sigma_derivative_text.y - VERTICAL_TEXT_SPACE * 1.5,
                                                   txt=a_text, fig=self.fig,
                                                   fontsize=self.sigma_derivative_text.fontsize)

        self.plt.pause(1)
        self.distribute_sigma_text_clone = FigText(x=self.sigma_clone_derivative_text.x, y=self.sigma_clone_derivative_text.y - VERTICAL_TEXT_SPACE * 1.5,
                                                   txt=b_text, fig=self.fig, fontsize=self.sigma_clone_derivative_text.fontsize)
        self.plt.pause(3)
        self.sigma_derivative_text.remove()
        self.sigma_clone_derivative_text.remove()

        self.distribute_sigma_text_clone.move(x=self.distribute_sigma_text_clone.x, y=self.distribute_sigma_text.y - VERTICAL_TEXT_SPACE * 2, total_frames=20, seconds=3)
        self.two_equations_two_unknowns = FigText(x=self.distribute_sigma_text.x, y=self.distribute_sigma_text_clone.y - VERTICAL_TEXT_SPACE * 2, txt="2 equations, 2 unknowns. Solve for A and B", fig=self.fig, wrap=True, fontsize=self.sigma_clone_derivative_text.fontsize)
        self.plt.pause(5)

    def show_solved_A_and_B(self):
        # Show solved equations for A and B in terms of sigma expressions
        distribute_sigma_text_location = (self.distribute_sigma_text.x, self.distribute_sigma_text.y)
        self.distribute_sigma_text.remove()
        self.distribute_sigma_text_clone.remove()
        self.two_equations_two_unknowns.remove()

        a_str = r"$A = \frac{{n\sum_{{i=1}}^nx_iy_i - \sum_{{i=1}}^nx_i\sum_{{i=1}}^ny_i}}{{n\sum_{{i=1}}^nx_i^2 - (\sum_{{i=1}}^nx_i)^2}}$"
        self.a_text = FigText(x=distribute_sigma_text_location[0], y=distribute_sigma_text_location[1] + VERTICAL_TEXT_SPACE * 2, txt=a_str, fig=self.fig, fontsize=20)
        b_str = r"$B = \frac{{\sum_{{i=1}}^ny_i}}{{n}} - A\frac{{\sum_{{i=1}}^nx_i}}{{n}}$"
        self.b_text = FigText(x=self.a_text.x, y=self.a_text.y - VERTICAL_TEXT_SPACE * 2, txt=b_str, fig=self.fig, fontsize=20)
        self.plt.pause(2)

    def evaluate_sigma_expressions(self):
        # Show how to compute each sigma expression in terms
        # of what the user inputted (ex: if they entered (1, 1), (2, 5), (3, 11); then the sum from i = 1 to n of x_i
        # is 1 + 1 + 1 = 3), do this sort of thing for all sigma expressions and plug them all back into the formula
        # that is solved for A and B, then show the result.

        # Display the strings of the sums of each sigma expression and then display the computed value

        sigma_xi_yi_str = ""
        sigma_xi_str = ""
        sigma_yi_str = ""
        sigma_xi_squared_str = ""
        sigma_xi_yi_value = 0
        sigma_xi_value = 0
        sigma_yi_value = 0
        sigma_xi_value_squared = 0
        n_value = len(self.x_coords)
        for i in range(len(self.x_coords)):
            sigma_xi_yi_str += f"{self.x_coords[i]}*{self.y_coords[i]} + "
            sigma_xi_str += f"{self.x_coords[i]} + "
            sigma_yi_str += f"{self.y_coords[i]} + "
            sigma_xi_squared_str += f"{self.x_coords[i]}^2 + "
            sigma_xi_yi_value += self.x_coords[i] * self.y_coords[i]
            sigma_xi_value += self.x_coords[i]
            sigma_yi_value += self.y_coords[i]
            sigma_xi_value_squared += self.x_coords[i]**2

            if i == len(self.x_coords) - 1:
                sigma_xi_yi_str = sigma_xi_yi_str[:-3]
                sigma_xi_str = sigma_xi_str[:-3]
                sigma_yi_str = sigma_yi_str[:-3]
                sigma_xi_squared_str = sigma_xi_squared_str[:-3]

        self.xi_yi_text = FigText(x=self.b_text.x, y=self.b_text.y - VERTICAL_TEXT_SPACE,
                                  txt=r"$\sum_{{i=1}}^nx_iy_i = $" + f"{sigma_xi_yi_str} = {sigma_xi_yi_value}", fig=self.fig, wrap=True, fontsize=self.b_text.fontsize * .75)
        self.plt.pause(2)
        self.xi_yi_text.set_text(r"$\sum_{{i=1}}^nx_iy_i = $" + str(sigma_xi_yi_value))
        self.plt.pause(1)

        self.xi_text= FigText(x=self.b_text.x, y=self.b_text.y - VERTICAL_TEXT_SPACE * 2,
                                  txt=r"$\sum_{{i=1}}^nx_i = $" + f"{sigma_xi_str} = {sigma_xi_value}",
                                  fig=self.fig, wrap=True, fontsize=self.b_text.fontsize * .75)
        self.plt.pause(2)
        self.xi_text.set_text(r"$\sum_{{i=1}}^nx_i = $" + str(sigma_xi_value))
        self.plt.pause(1)

        self.yi_text = FigText(x=self.b_text.x, y=self.b_text.y - VERTICAL_TEXT_SPACE * 3,
                                  txt=r"$\sum_{{i=1}}^ny_i = $" + f"{sigma_yi_str} = {sigma_yi_value}",
                                  fig=self.fig, wrap=True, fontsize=self.b_text.fontsize * .75)
        self.plt.pause(2)
        self.yi_text.set_text(r"$\sum_{{i=1}}^ny_i = $" + str(sigma_yi_value))
        self.plt.pause(1)

        self.xi_squared_text = FigText(x=self.b_text.x, y=self.b_text.y - VERTICAL_TEXT_SPACE * 4,
                                  txt=r"$\sum_{{i=1}}^nx_i^2 = $" + f"{sigma_xi_squared_str} = {sigma_xi_value_squared}",
                                  fig=self.fig, wrap=True, fontsize=self.b_text.fontsize * .75)
        self.plt.pause(2)
        self.xi_squared_text.set_text(r"$\sum_{{i=1}}^nx_i^2 = $" + str(sigma_xi_value_squared))
        self.plt.pause(1)

        self.n_text = FigText(x=self.b_text.x, y=self.b_text.y - VERTICAL_TEXT_SPACE * 5,
                                  txt=f"n = number of data points you entered = {n_value}",
                                  fig=self.fig, wrap=True, fontsize=self.b_text.fontsize * .75)
        self.plt.pause(2)
        self.n_text.set_text(f"n = {len(self.x_coords)}")
        self.plt.pause(1)

        # Substitute the expressions back into the formulas for a and b one by one

        self.a_text.set_text(r"$A = \frac{{n({xi_yi_text}) - \sum_{{i=1}}^nx_i\sum_{{i=1}}^ny_i}}{{n\sum_{{i=1}}^nx_i^2 - (\sum_{{i=1}}^nx_i)^2}}$"
                             .format(xi_yi_text=sigma_xi_yi_value))
        self.plt.pause(1)
        self.xi_yi_text.remove()
        self.plt.pause(1)

        self.a_text.set_text(r"$A = \frac{{n({xi_yi_text}) - ({xi_text})\sum_{{i=1}}^ny_i}}{{n\sum_{{i=1}}^nx_i^2 - ({xi_text})^2}}$"
                             .format(xi_yi_text=sigma_xi_yi_value, xi_text=sigma_xi_value))
        self.b_text.set_text(r"$B = \frac{{\sum_{{i=1}}^ny_i}}{{n}} - A\frac{{({xi_text})}}{{n}}$"
                             .format(xi_text=sigma_xi_value))
        self.plt.pause(1)
        self.xi_text.remove()
        self.plt.pause(1)

        self.a_text.set_text(r"$A = \frac{{n({xi_yi_text}) - ({xi_text})({yi_text})}}{{n\sum_{{i=1}}^nx_i^2 - ({xi_text})^2}}$"
                            .format(xi_yi_text=sigma_xi_yi_value, xi_text=sigma_xi_value, yi_text=sigma_yi_value))
        self.b_text.set_text(r"$B = \frac{{({yi_text})}}{{n}} - A\frac{{({xi_text})}}{{n}}$"
                             .format(xi_text=sigma_xi_value, yi_text=sigma_yi_value))
        self.plt.pause(1)
        self.yi_text.remove()
        self.plt.pause(1)

        self.a_text.set_text(r"$A = \frac{{n({xi_yi_text}) - ({xi_text})({yi_text})}}{{n({xi_squared_text}) - ({xi_text})^2}}$"
                            .format(xi_yi_text=sigma_xi_yi_value, xi_text=sigma_xi_value, yi_text=sigma_yi_value, xi_squared_text=sigma_xi_value_squared))
        self.plt.pause(1)
        self.xi_squared_text.remove()
        self.plt.pause(1)

        self.a_text.set_text(r"$A = \frac{{({n_text})({xi_yi_text}) - ({xi_text})({yi_text})}}{{({n_text})({xi_squared_text}) - ({xi_text})^2}}$"
                            .format(xi_yi_text=sigma_xi_yi_value, xi_text=sigma_xi_value, yi_text=sigma_yi_value,
                            xi_squared_text=sigma_xi_value_squared, n_text=n_value))
        self.b_text.set_text(r"$B = \frac{{({yi_text})}}{{({n_text})}} - A\frac{{({xi_text})}}{{({n_text})}}$"
                             .format(xi_text=sigma_xi_value, yi_text=sigma_yi_value, n_text=n_value))
        self.plt.pause(1)
        self.n_text.remove()
        self.plt.pause(1)

        self.calculated_a_value = (n_value * sigma_xi_yi_value - sigma_xi_value * sigma_yi_value) / (n_value * sigma_xi_value_squared - sigma_xi_value**2)
        self.calculated_a_value = round(self.calculated_a_value, 3)
        self.calculated_b_value = sigma_yi_value / n_value - self.calculated_a_value * (sigma_xi_value / n_value)
        self.calculated_b_value = round(self.calculated_b_value, 3)
        self.a_text.set_text(r"$A = \frac{{({n_text})({xi_yi_text}) - ({xi_text})({yi_text})}}{{({n_text})({xi_squared_text}) - ({xi_text})^2}} = {value}$"
                            .format(xi_yi_text=sigma_xi_yi_value, xi_text=sigma_xi_value, yi_text=sigma_yi_value,
                            xi_squared_text=sigma_xi_value_squared, n_text=n_value, value=self.calculated_a_value))
        self.plt.pause(2)
        self.b_text.set_text(r"$B = \frac{{({yi_text})}}{{({n_text})}} - ({a_value})\frac{{({xi_text})}}{{({n_text})}}$"
                             .format(xi_text=sigma_xi_value, yi_text=sigma_yi_value, n_text=n_value, a_value=self.calculated_a_value))
        self.plt.pause(2)
        self.b_text.set_text(r"$B = \frac{{({yi_text})}}{{({n_text})}} - ({a_value})\frac{{({xi_text})}}{{({n_text})}} = {value}$"
                            .format(xi_text=sigma_xi_value, yi_text=sigma_yi_value, n_text=n_value,
                                    a_value=self.calculated_a_value, value=self.calculated_b_value))
        self.plt.pause(2)
        self.a_text.set_text(f"A = {self.calculated_a_value}")
        self.b_text.set_text(f"B = {self.calculated_b_value}")

        self.plt.pause(2)
        self.regression_equation_text = FigText(x=self.b_text.x, y=self.b_text.x - VERTICAL_TEXT_SPACE * 2,
                                                txt=f"ŷ = {self.calculated_a_value}x + {self.calculated_b_value}",
                                                fig=self.fig, wrap=True, fontsize=self.b_text.fontsize)

    def animate_to_best_fit(self):
        # Zoom back out to what we had originally
        seconds = 3
        times_to_zoom = 20
        delay = seconds / times_to_zoom
        x_start_step, x_end_step = (self.original_x_lims[0] - self.ax.get_xlim()[0]) / times_to_zoom, \
                                   (self.original_x_lims[1] - self.ax.get_xlim()[1]) / times_to_zoom

        y_start_step, y_end_step = (self.original_y_lims[0] - self.ax.get_ylim()[0]) / times_to_zoom, \
                                   (self.original_y_lims[1] - self.ax.get_ylim()[1]) / times_to_zoom
        for _ in range(times_to_zoom):
            self.x_start += x_start_step
            self.x_end += x_end_step
            self.y_start += y_start_step
            self.y_end += y_end_step
            self.plt.pause(delay)
            self.ax.set_xlim(self.x_start, self.x_end)
            self.ax.set_ylim(self.y_start, self.y_end)

        # First, remove the to become best fit line, the residual lines, and circle previously drawn
        for line in self.lines:
            line.remove()
        for line in self.residual_lines:
            line.remove()
        self.circle.remove()

        # Then, update our line until it reaches our best fit equation
        seconds_to_draw = 5
        next_a, next_b = self.to_become_y_fit_coefficients[0], self.to_become_y_fit_coefficients[1]
        a_step = (self.calculated_a_value - self.to_become_y_fit_coefficients[0]) / TIMES_TO_DRAW
        b_step = (self.calculated_b_value - self.to_become_y_fit_coefficients[1]) / TIMES_TO_DRAW
        for i in range(TIMES_TO_DRAW):
            next_a += a_step
            next_b += b_step
            self.fit_eq_text.set_text(f"ŷ = {round(next_a, 3)}x + {round(next_b, 3)}")
            y_values = np.polyval([next_a, next_b], self.x_continuous)
            line_x_to_plot = []
            line_y_to_plot = []

            for j in range(TIMES_TO_DRAW):
                line_x_to_plot.append(self.x_continuous[j])
                line_y_to_plot.append(y_values[j])
            lin_eq_line = self.ax.plot(line_x_to_plot, line_y_to_plot, color='g', alpha=0.5)[0]

            # Update the residuals accordingly
            residual_x_to_plot = []
            residual_y_to_plot = []
            self.residual_line_info = [[(x, y), (x, np.polyval([next_a, next_b], x))] for x, y in
                                       zip(self.x_coords, self.y_coords)]
            self.residual_lines = [] # Store references for all the residual lines plotted here

            for line in self.residual_line_info:
                x = line[0][0]
                y = line[0][1]
                y_predict = line[1][1]
                y_continuous = np.linspace(y_predict, y, num=TIMES_TO_DRAW)
                for y in y_continuous:
                    residual_x_to_plot.append(x)
                    residual_y_to_plot.append(y)

                residual_line = self.ax.plot(residual_x_to_plot, residual_y_to_plot, color='r', alpha=0.5)[0]
                self.residual_lines.append(residual_line)
                residual_x_to_plot.clear()
                residual_y_to_plot.clear()

            self.plt.pause(seconds_to_draw / TIMES_TO_DRAW)
            if i != TIMES_TO_DRAW - 1:
                lin_eq_line.remove()
                for line in self.residual_lines:
                    line.remove()

        self.plt.pause(5)

    def thanks_for_watching(self):
        FigText(x=.25, y=.5, txt="Thanks for watching!!!", fig=self.fig, wrap=False, fontsize=50)
        self.plt.pause(10)

    def update(self, frame):
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            step()
            self.current_step += 1

    def run(self):
        animation = FuncAnimation(self.fig, func=self.update, frames=len(self.steps), interval=10)
        self.plt.show()
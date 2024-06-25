# Linear regression visualizer will show the user a process visually of how to fit
# a linear equation in general form of their choice to a certain number of data points they choose

# Project general outline:
# 1. Prompt the user for a number of data points in a certain range
# 3. Display a window of their selected data points for each one they select
# 4. Show the user their data point coordinates: (X1, Y1) (Xi, Yi) (Xn, Yn) where n is the last data point
# 5. Show some linear polynomial that somewhat fits the data but not exactly with the equation of it
# 6. Show the residuals and zoom in on the main equation for the ith residual which is in absolute value
# 7. Then, expand the window for the animated calculations to be done
# 8. Tell user we need to minimize sum of the errors squared to get the best fit polynomial in new window. Don't
# minimize sum of absolute value of errors since the derivative of abs(exp) wouldn't have a derivative of zero due to sharp turn
# 9. Show sum of squared errors and then substitute the residual with its equation
# 10. Say we must minimize the sum of the residuals by taking the partial derivative WRT each variable
# 11. Circle each variable and then animate them going from the polynomial to d/d(variable) of sum(residuals)^2
# 12. Animate all of the derivatives being taken and say we need to solve for "A" and "B"
# 13. Say we are solving the system of equations and then come up with each of the variable values through
#     using a library to do this for you
# 14. Erase everything except variables' results and plug them into sum and show the value of the sum
# 15. On window with graph (if space permits, otherwise do on window for calculations):
# Make variables approach their values to minimize this function over time and show new graph for each decrease and new sum of residuals and new residuals
# until we reach the best fit curve (which is just substituting each variable of the polynomial with the
# ones we attained through differentiating and solving system of 2 equations)
# 16. Cool thank you for watching at the end.

import matplotlib.pyplot as plt

from handle_data_entry import handle_data_entry
from main_animation import MainAnimation

WIDTH, HEIGHT = 12, 6
MIN_POINTS, MAX_POINTS = 3, 6 # User can enter between 3-6 data points (inclusive)

if __name__ == "__main__":
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))
    ax2.set_visible(False)
    fig.subplots_adjust(bottom=0.2)

    x_coords, y_coords = handle_data_entry(fig, ax, plt, MIN_POINTS, MAX_POINTS)

    # Play the main animation
    main_animation = MainAnimation(plt, ax, ax2, fig, x_coords, y_coords)
    main_animation.run()
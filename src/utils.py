from IPython.display import display, Markdown

def print_bold(text):
    display(Markdown(f"**{text}**"))

# Function to color values based on skewness and kurtosis
def color_value(val):
    if val > 1 or val < -1:
        return f"\033[91m{val:.2f}\033[0m"  # Red color for values above 1 or below -1
    return f"{val:.2f}"
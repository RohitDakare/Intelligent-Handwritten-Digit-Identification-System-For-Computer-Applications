from tkinter import *
import numpy as np
from PIL import ImageGrab
from Prediction import predict

window = Tk()
window.title("Handwritten Digit Recognition")
l1 = Label(window, text="", font=('Algerian', 20))
l1.place(x=230, y=420)

def MyProject():
    global l1
    # Setting co-ordinates of canvas
    widget = cv
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Capture canvas and resize to (28x28)
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    x = np.asarray(img)
    vec = x.flatten().reshape(1, 784)  # Flatten image into a vector of 784 (28x28)

    # Load trained Theta values
    Theta1 = np.loadtxt('Theta1.txt')
    Theta2 = np.loadtxt('Theta2.txt')

    # Make prediction
    pred = predict(Theta1, Theta2, vec / 255.0)

    # Display result
    l1.config(text="Digit = " + str(pred[0]))

lastx, lasty = None, None

# Clear canvas
def clear_widget():
    global l1
    cv.delete("all")
    l1.config(text="")

# Start drawing on canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

# Draw lines on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

# Setup GUI
Label(window, text="Handwritten Digit Recognition", font=('Algerian', 25), fg="blue").place(x=35, y=10)
Button(window, text="1. Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_widget).place(x=120, y=370)
Button(window, text="2. Predict", font=('Algerian', 15), bg="white", fg="red", command=MyProject).place(x=320, y=370)

# Canvas for drawing
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)
cv.bind('<Button-1>', event_activation)

# Window size
window.geometry("600x500")
window.mainloop()
#rd
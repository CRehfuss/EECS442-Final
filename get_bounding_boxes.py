## Claire Rehfuss
## Made for EECS 442 - Computer Vision
import matplotlib
import Tkinter as tk
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from Tkinter import *
from tkFileDialog import *
from scipy.misc import imread
from scipy.misc import imresize
from matplotlib.widgets import Button
from predict import predict_char
from PIL import Image
import numpy as np
import math
from googletrans import Translator

def get_label_name(label):
    if label < 10:
        return str(label)
    if label < 36:
        return chr(ord('A') - 10 + label)
    return chr(ord('a') - 36 + label)

class Annotate(object):
    def __init__(self, uploaded_image):
        self.uploaded_image = uploaded_image
        plt.imshow(uploaded_image, zorder = 0)
        self.ax = plt.gca()
        axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
        self.bsubmit = Button(axcut, "Submit", color = 'red', hovercolor = "green")
        self.rect = Rectangle((0,0), 1, 1, fc = (1,0,0,.5), ec = (0,0,0,1), lw=2)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.boundedboxes = []
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.bsubmit.on_clicked(self.submit)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    # For drawing the boxes. When release it saves the coordinates in list of list datastructure
    def on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.boundedboxes.append([self.x0,self.y0,self.x1,self.y1])
        self.ax.figure.canvas.draw()

    # On click of submit button
    # Needs to be updated to send these values to the function which will
    # recognize the objects
    # Then closes all the matplotlib objects that are open
    def submit(self,event):

        word = ""
        count = 0
        cropped_images = np.empty(shape=(4, 32, 32, 3))
        for i in self.boundedboxes:
            if i[1]>1:
                cropped_images[count,:,:,:] = crop_image(self.uploaded_image, i)
                count +=1

        guesses = predict_char(cropped_images)
        for i in guesses:
            word += get_label_name(i)

        Translate(word)

# run "pip install googletrans"
class Translate(object):
    def __init__(self,word):
        self.word = word
        self.root = tk.Tk()
        # use width x height + x_offset + y_offset (no spaces!)
        self.root.geometry("%dx%d+%d+%d" % (330, 80, 200, 150))
        self.root.title("tk.Optionmenu as combobox")
        self.var = tk.StringVar(self.root)
        self.language = "English"
        # initial value
        self.var.set('Spanish')
        self.choices = ['Spanish', 'French', 'Dutch', 'Russian', 'German', 'Thai']
        self.dictionary = ['es', 'fr', 'nl', 'ru', 'de', 'th']
        option = tk.OptionMenu(self.root, self.var, *self.choices)
        option.pack(side='left', padx=10, pady=10)
        button = tk.Button(self.root, text="check value slected", command=self.select)
        button.pack(side='left', padx=20, pady=10)
        self.root.mainloop()

    def select(self,):
        sf = "value is %s" % self.var.get()
        self.root.title(sf)
        # optional
        self.language = self.var.get()
        self.translate_word()

    def translate_word(self,):
        translator = Translator()
        i = self.choices.index(self.language)
        translation = translator.translate(self.word, dest=self.dictionary[i], src='en')
        print translation.origin + " -> " +translation.text

def crop_image(image, x_y_values):
    cropped_img = image[int(math.ceil(x_y_values[1])):int(math.ceil(x_y_values[3])), int(math.ceil(x_y_values[0])):int(math.ceil(x_y_values[2])), :]
    cropped_img = imresize(cropped_img, (32, 32))
    cropped_img = cropped_img[:, :, :3]
    return cropped_img

def main():
    """Create our Master class instantiation."""
    #Allows a user to open an image from their computer
    root = Tk()
    root.withdraw()
    filename = askopenfilename()
    root.update()
    root.destroy()
    img = imread(filename)

    a = Annotate(img)
    plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()

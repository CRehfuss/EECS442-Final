## Claire Rehfuss
## Made for EECS 442 - Computer Vision
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.misc import imread
from matplotlib.widgets import Button

class Annotate(object):
    def __init__(self, uploaded_image):
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
        print(self.boundedboxes)
        #plt.close('all')

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

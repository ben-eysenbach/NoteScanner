import Image
import ImageTk
import Tkinter

class BaseImage:

    def __init__(self, img):
        self.corners = []
        self.root = Tkinter.Tk()
        width, height = img.size
        self.canvas = Tkinter.Canvas(self.root, width=width, height=height)

        tkpi = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=Tkinter.NW, image=tkpi)
        self.canvas.bind('<Button>', self.on_click)
        self.canvas.pack()
        self.root.mainloop()

    def on_click(self, event):
        print 'Click:', event.x, event.y
        if len(self.corners) < 4:
            self.canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, outline="red",
                fill="green", width=2)
            self.corners.append((event.x, event.y))
        if len(self.corners) == 4:
            self.root.quit()





if __name__ == '__main__':
    img = Image.open('flag.jpg')
    bi = BaseImage(img)
    print bi.corners

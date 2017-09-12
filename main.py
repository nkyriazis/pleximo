import os
import wx
import wx.xrc
import wx.html
import numpy as np
from scipy.misc import imread, imresize
import matplotlib
from PIL.Image import fromarray
from itertools import groupby
from tempfile import NamedTemporaryFile

matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar


class ImagePleximatization:
    def __init__(self, onLoad=None, onResize=None, onRecolor=None, onMargin=None):
        self.onLoad = onLoad
        self.onResize = onResize
        self.onRecolor = onRecolor
        self.onMargin = onMargin

        self.image = None
        self.originalHeight = None
        self.originalWidth = None
        self.width = None
        self.height = None
        self.numberOfColors = None

        self.top = 0
        self.bottom = 0
        self.left = 0
        self.right = 0

    def load(self, url):
        self.image = imread(url)
        self.originalHeight, self.originalWidth, d = self.image.shape
        self.height = self.originalHeight
        self.width = self.originalWidth
        self.numberOfColors = len(self.computeUniqueColors(self.image))
        if self.onLoad: self.onLoad(self)

    def computeImage(self):
        H, W, D = self.image.shape
        pilImage = fromarray(self.image)
        pilImage = pilImage.resize((self.width, self.height))
        pilImage = pilImage.quantize(self.numberOfColors).convert()
        ret = np.asarray(pilImage)

        computePad = lambda v: -v if v < 0 else 0
        topPad = computePad(self.top)
        bottomPad = computePad(self.bottom)
        leftPad = computePad(self.left)
        rightPad = computePad(self.right)

        ret = np.pad(ret, ((topPad, bottomPad), (leftPad, rightPad), (0, 0)), 'edge')

        computeSelect = lambda v: v if v > 0 else 0
        topSelect = computeSelect(self.top)
        bottomSelect = computeSelect(self.bottom)
        leftSelect = computeSelect(self.left)
        rightSelect = computeSelect(self.right)

        ret = ret[topSelect:self.height - bottomSelect, leftSelect:self.width - rightSelect, :]

        return ret

    def computeUniqueColors(self, image=None):
        if image is None:
            image = self.computeImage()
        return fromarray(image).getcolors()

    def resize(self, newsize):
        self.height, self.width = newsize
        if self.onResize: self.onResize(self)

    def setMargin(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        if self.onMargin: self.onMargin(self)

    def recolor(self, numberOfColors):
        self.numberOfColors = numberOfColors
        if self.onRecolor: self.onRecolor(self)

    def generateTrajectory(self):
        trajectory = []
        for i, line in enumerate(xrange(1, self.width + self.height)):
            start_col = max(0, line - self.height)
            count = min(line, (self.width - start_col), self.height)
            traversal = []
            for j in xrange(count):
                traversal.append((i, min(self.height, line) - j - 1, start_col + j))
            trajectory += traversal[::-1 if i % 2 else 1]
        return trajectory

    def generateInstructions(self):
        img = self.computeImage()
        H, W, _ = img.shape
        tr = np.asarray(self.generateTrajectory())
        colors = map(tuple, img[tr[:, 1], tr[:, 2]].tolist())

        stops = np.zeros(len(colors), dtype=np.bool)
        for diag, ((i, j), (k, l)) in enumerate(zip(tr[:-1, 1:], tr[1:, 1:])):
            if (i != H - 1 and k == H - 1) or (j != W - 1 and l == W - 1):
                stops[diag + 1] = 1

        instructions = '<font size=14><table width="100%">'
        for i, (key, what) in enumerate(groupby(zip(tr, colors, stops), lambda x: x[0][0])):
            color = '#FFFFFF' if i % 2 else '#DDDDDD'
            line = '<tr><td bgcolor="%s">' % color
            line += '%d %s' % (key + 1, '&uarr;' if i % 2 == 0 else '&darr;')
            for j, (key2, what2) in enumerate(groupby(what, lambda x: x[1])):
                what2 = list(what2)
                stop = what2[-1][-1]
                line += ' %d <span style="color:rgb(%d, %d, %d)"><b>&#9632;</b></span>' % (
                len(what2), key2[0], key2[1], key2[2])
                if stop: line += ' stop'
            line += '</td></tr>'
            instructions += line
        instructions += '</table></font>'

        return tr, stops, colors, instructions


class PleximoApp(wx.App):
    def OnInit(self):
        self.res = wx.xrc.XmlResource('pleximo.xrc')
        self.init_frame()
        wx.FileSystem.AddHandler(wx.MemoryFSHandler())
        self.imgRAM = wx.MemoryFSHandler()
        self.hasPng = False
        return True

    def UpdateViews(self, image):
        self.frame.Freeze()
        updateInput = lambda txt, v: txt.ChangeValue(str(v))
        updateInput(self.input_width, image.width)
        updateInput(self.input_height, image.height)
        updateInput(self.input_colors, image.numberOfColors)
        updateInput(self.input_top_margin, image.top)
        updateInput(self.input_bottom_margin, image.bottom)
        updateInput(self.input_left_margin, image.left)
        updateInput(self.input_right_margin, image.right)

        img = image.computeImage()
        self.ax.clear()
        self.ax.set_title('Pleximo instructions')
        self.ax.imshow(img)
        # if self.image.width < 100 and self.image.height < 100:
        #     self.ax.grid()
        #     self.ax.set_xticks(np.linspace(1, self.image.width, self.image.width + 1) - 0.5)
        #     self.ax.set_yticks(np.linspace(1, self.image.height, self.image.height + 1) - 0.5)
        #     self.ax.set_xticklabels([])
        #     self.ax.set_yticklabels([])

        H = self.image.height
        W = self.image.width

        instructions = ''
        if W < 100 and H < 100:
            tr, stops, colors, instructions = self.image.generateInstructions()
            for i, (key, idx) in enumerate(groupby(xrange(len(colors)), lambda i: colors[i])):
                color = 'g' if i % 2 else 'b'
                style = '%s-o' % color
                idxs = np.asarray(list(idx))
                self.ax.plot(tr[idxs, 2], tr[idxs, 1], style, linewidth=1, alpha=0.5, markersize=2)
            self.ax.scatter(tr[stops, 2], tr[stops, 1], s=50, alpha=1, c=[1, 0, 1])

        self.figure.canvas.draw()
        self.frame.Thaw()

        W, H = self.figure.canvas.get_width_height()
        bmp = wx.EmptyImage(W, H)
        bmp.SetData(self.figure.canvas.tostring_rgb())
        bmp = wx.BitmapFromImage(bmp)
        if self.hasPng:
            self.imgRAM.RemoveFile('im.png')
        self.imgRAM.AddFile('im.png', bmp, wx.BITMAP_TYPE_PNG)
        self.hasPng = True
        # instructions = '<img width="100%" src="memory:im.png"/>' + instructions
        # self.html_instructions.SetPage(instructions)

    def OnImageLoad(self, image):
        self.UpdateViews(image)

    def OnImageResize(self, image):
        self.UpdateViews(image)

    def OnImagerRecolor(self, image):
        self.UpdateViews(image)

    def init_frame(self):
        self.image = ImagePleximatization(onLoad=self.OnImageLoad,
                                          onResize=self.OnImageResize,
                                          onRecolor=self.OnImagerRecolor,
                                          onMargin=self.OnImageMargin)
        self.frame = self.res.LoadFrame(None, 'main_frame')
        self.printer = wx.html.HtmlEasyPrinting(parentWindow=self.frame)

        # self.frame_instructions = self.res.LoadFrame(self.frame, 'frame_instructions')
        # self.html_instructions = wx.html.HtmlWindow(self.frame_instructions)
        # self.res.AttachUnknownControl('html_instructions', self.html_instructions, self.frame_instructions)

        self.controls = wx.xrc.XRCCTRL(self.frame, 'controls')
        self.figure = Figure()
        self.plot = FigCanvas(self.frame, wx.ID_ANY, self.figure)
        self.res.AttachUnknownControl('plot_view', self.plot, self.frame)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.ax.set_aspect('equal')

        getInput = lambda x: wx.xrc.XRCCTRL(self.frame, x)
        self.input_width = getInput('input_width')
        self.input_height = getInput('input_height')
        self.input_colors = getInput('input_colors')
        self.input_top_margin = getInput('input_top_margin')
        self.input_bottom_margin = getInput('input_bottom_margin')
        self.input_left_margin = getInput('input_left_margin')
        self.input_right_margin = getInput('input_right_margin')

        self.button_reset = wx.xrc.XRCCTRL(self.frame, 'button_reset')

        self.frame.Bind(wx.EVT_MENU, self.OnExit, id=wx.xrc.XRCID('menu_exit'))
        self.frame.Bind(wx.EVT_MENU, self.OnLoadImage, id=wx.xrc.XRCID('menu_load_image'))
        self.frame.Bind(wx.EVT_MENU, self.OnPrint, id=wx.xrc.XRCID('menu_print'))

        self.frame.Bind(wx.EVT_TEXT_ENTER, self.UpdateSize, self.input_width)
        self.frame.Bind(wx.EVT_TEXT_ENTER, self.UpdateSize, self.input_height)
        self.frame.Bind(wx.EVT_TEXT_ENTER, self.UpdateColors, self.input_colors)

        self.frame.Bind(wx.EVT_TEXT_ENTER, self.OnChangeMargin, self.input_top_margin)
        self.frame.Bind(wx.EVT_TEXT_ENTER, self.OnChangeMargin, self.input_bottom_margin)
        self.frame.Bind(wx.EVT_TEXT_ENTER, self.OnChangeMargin, self.input_left_margin)
        self.frame.Bind(wx.EVT_TEXT_ENTER, self.OnChangeMargin, self.input_right_margin)

        self.frame.Bind(wx.EVT_BUTTON, self.OnReset, self.button_reset)
        self.frame.Show()
        # self.frame_instructions.Show()

    def OnPrint(self, evt):
        self.printer.PreviewText('<img width="100%" src="memory:im.png"/>' + self.image.generateInstructions()[-1])

    def OnChangeMargin(self, evt):
        get = lambda x: int(x.GetValue())
        self.image.setMargin(get(self.input_top_margin),
                             get(self.input_bottom_margin),
                             get(self.input_left_margin),
                             get(self.input_right_margin))

    def OnReset(self, evt):
        self.frame.Freeze()
        numColors = len(self.image.computeUniqueColors(self.image.image))
        self.image.resize((self.image.originalHeight, self.image.originalWidth))
        self.image.recolor(numColors)
        self.image.setMargin(0, 0, 0, 0)
        self.frame.Thaw()

    def UpdateSize(self, evt):
        width = int(self.input_width.GetValue())
        height = int(self.input_height.GetValue())
        self.image.resize((height, width))

    def UpdateColors(self, evt):
        self.image.recolor(int(self.input_colors.GetValue()))

    def OnExit(self, evt):
        self.frame.Destroy()

    def OnImageMargin(self, image):
        self.UpdateViews(image)

    def OnLoadImage(self, evt):
        file_open = wx.FileDialog(self.frame, 'Load image', wildcard="Image files (*.png)|*.png",
                                  style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if file_open.ShowModal() == wx.ID_OK:
            url = str(file_open.GetPath())
            self.image.load(url)


if __name__ == '__main__':
    app = PleximoApp()
app.MainLoop()

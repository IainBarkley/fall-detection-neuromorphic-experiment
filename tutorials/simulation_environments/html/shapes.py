try:
    import numpy as np
    from PIL import Image
    import base64
    from io import BytesIO
except:
    print("Warning: Could not import libraries for image function")

def circle(x=0, y=0, r=.3, color='purple'):
    return '<circle cx="{0}" cy="{1}" r="{2}" fill="{3}"/>'.format(x, y, r, color)

def rectangle(x=0, y=0, width=1, height=1, color='white', outline_color='black', outline_width=.1):
    return '<rect width="{0}" height="{1}" style="fill:{2};stroke:{3};stroke-width:{4}" transform="translate({5},{6})"/>'.format(
        width, height, color, outline_color, outline_width, x, y,
    )

def triangle(x=0, y=0, th=0, scale=1, color='purple'):
    return '<polygon points="0.25,0.25 -0.25,0.25 0,-0.5" style="fill:{0}" transform="translate({1},{2}) rotate({3}) scale({4}, {4})"/>'.format(
        color, x, y, th, scale,
    )

def line(x1, y1, x2, y2, color='black', width=.1):
    return '<line x1="{0}" y1="{1}" x2="{2}" y2="{3}" style="stroke:{4};stroke-width:{5}"/>'.format(
        x1, y1, x2, y2, color, width,
    )

def image(img, x=0, y=0, th=0, scale=1, height=10, width=10, cmap=None):

    if type(img) == str:
        png = Image.open(img)
    else:
        if cmap is not None:
            values = (cmap(img)*255).astype(np.uint8)
        else:
            values = (img*255).astype(np.uint8)
        png = Image.fromarray(values)
    buffer = BytesIO()
    png.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return '<image width="{0}" height="{1}" xlink:href="data:image/png;base64,{2}" style="image-rendering: pixelated;" transform="translate({3},{4}) rotate({5}) scale({6}, {6})"/>'.format(height, width, ''.join(img_str), x, y, th, scale)

def canvas(shape_list, width=10, height=10):
    svg = '<svg width="100%%" height="100%%" viewbox="0 0 {0} {1}">'.format(width, height)
    for shape in shape_list:
        svg += shape
    svg += '</svg>'
    return svg


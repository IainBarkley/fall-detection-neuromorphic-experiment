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

def canvas(shape_list, width=10, height=10):
    svg = '<svg width="100%%" height="100%%" viewbox="0 0 {0} {1}">'.format(width, height)
    for shape in shape_list:
        svg += shape
    svg += '</svg>'
    return svg


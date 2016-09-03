import pngcanvas


def create_png(r, g, b):
    """Create a PNG from r, g, b, data.

    RGB should be on the [-1.0, 1.0] scale.

    Returns:
        bytes: PNG image bytes
    """
    r = round((r + 1) / 2 * 255)
    g = round((g + 1) / 2 * 255)
    b = round((b + 1) / 2 * 255)

    canvas = pngcanvas.PNGCanvas(150, 150, bgcolor=(r, g, b, 0xff))
    data = canvas.dump()
    return data

import requests
from colorbot.data import Color, hex_to_rgb


def valspar():
    url = "http://www.valsparpaint.com/vservice/json/colors/color-wall?localeId=1001&channelId=1001"

    result = requests.get(url)
    data = result.json()

    for family in data["data"]:
        for color in family["colors"]:
            name = color["name"].lower()
            r = color["rgb"]["r"]
            g = color["rgb"]["g"]
            b = color["rgb"]["b"]

            r_val = 2 * r / 255 - 1.0
            g_val = 2 * g / 255 - 1.0
            b_val = 2 * b / 255 - 1.0

            c = Color(name, r_val, g_val, b_val)
            yield c


def sherwin_williams():
    url = "http://www.sherwin-williams.com/homeowners/color/find-and-explore-colors/paint-colors-by-family/json/full/"
    result = requests.get(url)
    data = result.json()

    for family_name, family in data.items():
        for color in family["items"]:
            hex = color["attributes"]["data-color-hex"]
            name_field = color["attributes"]["data-search-by"]

            name, sep, extra = name_field.partition("|")
            name = name.lower()

            r, g, b = hex_to_rgb(hex)

            c = Color(name, r, g, b)
            yield c


color_sources = [
    valspar,
    sherwin_williams,
]

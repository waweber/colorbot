import json
import re

import requests

from colorbot import constants
from colorbot.data import Color, hex_to_rgb
import html


def valspar():
    url = "http://www.valsparpaint.com/vservice/json/colors/color-wall?localeId=1001&channelId=1001"

    result = requests.get(url)
    data = result.json()

    seen_ids = set()

    for family in data["data"]:
        for color in family["colors"]:
            if color["id"] not in seen_ids:
                name = color["name"].lower()
                r = color["rgb"]["r"]
                g = color["rgb"]["g"]
                b = color["rgb"]["b"]

                r_val = 2 * r / 255 - 1.0
                g_val = 2 * g / 255 - 1.0
                b_val = 2 * b / 255 - 1.0

                name_fmt = "%s%s%s" % (
                    constants.START_SYMBOL,
                    name,
                    constants.END_SYMBOL,
                )

                seen_ids.add(color["id"])

                c = Color(name_fmt, r_val, g_val, b_val)
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

            name_fmt = "%s%s%s" % (
                constants.START_SYMBOL,
                name,
                constants.END_SYMBOL,
            )

            c = Color(name_fmt, r, g, b)
            yield c


def behr():
    url = "http://www.behr.com/mainService/services/colornx/all.js"

    response = requests.get(url)
    data_str = response.content.decode("utf-8")[15:-1]

    data = json.loads(data_str)

    seen_names = set()

    for color in data[1:]:
        name = color[0].lower()

        if name not in seen_names:
            rgb_txt = color[3]
            rgb = hex_to_rgb(rgb_txt[1:])

            name_fmt = "%s%s%s" % (
                constants.START_SYMBOL,
                name,
                constants.END_SYMBOL,
            )

            seen_names.add(name)

            yield Color(name_fmt, *rgb)


def benjamin_moore():
    url = "http://67.222.214.23/bmServices/ColorExplorer/colorexplorer.svc/Colors_GetByFilter?locale=en_US&collectionCode=&familyCode=&trendCode="

    response = requests.get(url)
    data = response.json()

    for color in data:
        name = color["colorName"].lower()
        r = 2.0 * color["RGB"]["R"] / 255 - 1.0
        g = 2.0 * color["RGB"]["G"] / 255 - 1.0
        b = 2.0 * color["RGB"]["B"] / 255 - 1.0

        name_fmt = "%s%s%s" % (
            constants.START_SYMBOL,
            name,
            constants.END_SYMBOL,
        )

        yield Color(name_fmt, r, g, b)


def dulux():
    url = "https://www.dulux.co.uk/en/api/products/colors"

    result = requests.get(url)

    data = result.json()

    for color in data["colors"]:
        name = color["name"].lower()
        hex = color["rgb"]

        r, g, b, = hex_to_rgb(hex)

        name_fmt = "%s%s%s" % (
            constants.START_SYMBOL,
            name,
            constants.END_SYMBOL,
        )

        yield Color(name_fmt, r, g, b)


def ppg():
    url = "https://pittsburghpaintsandstains.com/color/paint-colors"

    result = requests.get(url)

    page = result.content.decode("utf-8")

    pattern = """<a href=".*?" style="background-color:rgb\\(([0-9]+), ([0-9]+), ([0-9]+)\\);" .*? title="(.*?) &ndash; .*?>"""

    matches = re.finditer(pattern, page)

    for match in matches:
        r = int(match.group(1)) / 255
        g = int(match.group(2)) / 255
        b = int(match.group(3)) / 255

        name = html.unescape(match.group(4)).lower().strip()

        name_fmt = "%s%s%s" % (
            constants.START_SYMBOL,
            name,
            constants.END_SYMBOL,
        )

        yield Color(name_fmt, r, g, b)


def colorhexa():
    url = "http://www.colorhexa.com/color-names"

    result = requests.get(url)
    page = result.content.decode("utf-8")

    # parsing HTML with regex, grumble grumble
    pattern = """<a class="t." href="/([a-z0-9]{6})">(.*?)</a></td>"""

    matches = re.finditer(pattern, page)

    for match in matches:
        hex = match.group(1)
        name = html.unescape(match.group(2)).strip().lower()

        r, g, b = hex_to_rgb(hex)

        name_fmt = "%s%s%s" % (
            constants.START_SYMBOL,
            name,
            constants.END_SYMBOL,
        )

        yield Color(name_fmt, r, g, b)


color_sources = [
    valspar,
    sherwin_williams,
    behr,
    benjamin_moore,
    dulux,
    ppg,
    colorhexa,
]

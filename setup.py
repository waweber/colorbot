from setuptools import setup, find_packages

reqs = [
    "tensorflow",
    "numpy",
    "requests",
    "tweepy",
    "pngcanvas",
]

setup(
    name="colorbot",
    version="1.0.0",
    packages=find_packages(),

    install_requires=reqs,

    entry_points={
        "console_scripts": [
            "colorbot_prepare = colorbot.scripts.prepare:prepare_data",
            "colorbot_train = colorbot.scripts.train:train",
            "colorbot_sample = colorbot.scripts.sample:sample",
            "colorbot_auth = colorbot.scripts.auth:auth",
            "colorbot_post = colorbot.scripts.post:post",
            "colorbot_run = colorbot.scripts.bot:bot",
        ],
    },
)

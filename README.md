# colorbot

A bot that learns how to map names to colors and colors to names, and provides
this as a service via a Twitter bot.

## Overview

This is a machine learning program that uses Google's TensorFlow. It scrapes
color information from various sources, e.g. paint companies like Behr and
Valspar, then trains two recurrent neural network (RNN) models. One is an
encoder, which takes a sequence of characters and outputs an RGB vector, and the
other is a decoder, which takes an RGB vector and outputs a sequence of
characters. These models probably get horribly overfitted, which is fine.

Once the models are fitted, it runs a twitter bot using Tweepy that listens for
mentions. Given a hex color code, it replies with a name for the color. Given
a text string, it replies with the hex code for the color and an example image.

## Usage

After installing, use `colorbot_auth` to get OAuth credentials for the Twitter
account (you'll need a client key and secret from Twitter).

Use `colorbot_prepare` to scrape color information and build a character
vocabulary.

Now, you're ready to run `colorbot_train` on this data. Let it run until the
loss of both models stops decreasing (or however long you like). Check out the
`-n` option to increase or decrease the hidden size to experiment. Use Ctrl-C
to stop the training.

You can get a sample output HTML page of colors with `colorbot_sample`. Make
sure to use the same `-n` as training if you changed it. The output shows a
random color, then the generated name for the color, then the color generated
from that name.

`colorbot_post` generates a random color, names it, and uploads an example to
Twitter. You'll need the credential file you saved from `colorbot_auth`.

`colorbot_run` runs the bot that waits for mentions and replies. You should use
this with a program that manages daemons, e.g. supervisor.

## Notes and Observations

- The models aren't *good*, since there is not a significant underlying function
mapping a color to a name and vice-versa. I.e., a color doesn't encode enough
information to get much of a name out of. Thus, the best usage for this is a
humorous Twitter bot.

- The encoder doesn't tend to produce very saturated colors. This is probably
due to the fact that people don't tend to paint interiors with fluorescent
orange or other bright colors, and the majority of the data set comes from
interior paint colors. If many more colors from other sources were to be added,
this would likely change.

- Both models need to be loaded (and in the same order) to be able to restore
their parameters. This can probably be fixed by specifying the variables to use
with `tf.train.Saver`.

- The mention watching and replying functionalities run in separate threads. I
implemented a work queue; the main, stream reading thread reads tweets and puts
tasks into a list, and a separate worker thread processes them. This isn't super
necessary, I just wanted to practice threading/synchronization. For throttling,
the worker thread just sleeps; this can probably be improved with a
`threading.Timer`.

    - Tweepy's stream listener has to be run synchronously in the main thread.
    This is because there's no (public) way to interrupt its blocking read. I
    get out of it with a KeyboardInterrupt raised from SIGINT or SIGTERM.
    
- There is no command line logging configuration. I hope you like debug
messages.
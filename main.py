# THIS PROGRAM IS TO BE USED FOR EDUCATIONAL PURPOSES ONLY.
# CAN BE USED FOR INTERNAL PEN-TESTING, STAFF RECRUITMENT, SOCIAL ENGAGEMENT

import time
import requests
import json
import collections
import random
import tweepy
import credentials
import markovify
import argparse
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import numpy
import post_status
from cluster import *

STATUS_MAX_LEN = 140
NN_SAMPLE_COMMAND = "./sample.sh"
SECONDS_PER_TIMELINE = 5
SECONDS_TO_POST = 8
CREATED_AT_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"
MAX_TIMELINE_POSTS = 200


def is_target(screen_name, disable_targeting, model_file='cluster.pkl'):
    """
    Returns a boolean for whether the user should be selected according
    to label identity returned by a prediction from a pretrained
    clustering algorithm.
    """
    if disable_targeting:
        return True
    else:
        auth = tweepy.OAuthHandler(credentials.consumer_key,
                                   credentials.consumer_secret)
        auth.set_access_token(credentials.access_token,
                              credentials.access_token_secret)
        api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
        user_array = numpy.array([api.get_user(screen_name=screen_name)])
        model = joblib.load(model_file)
        cluster_label = model.predict(user_array)
        return cluster_label == 1


def preprocess_post(post):
    processed_post_text = []
    for word in post["text"].split(" "):
        if (
                len(word) > 0 and       # Remove empty strings
                word[0] != '@' and      # Remove at mentions and usernames
                word[0] != '/' and      # Remove emojis and some weird stuff
                "http" not in word and  # Remove links
                "RT" not in word):      # Remove RTs
            processed_post_text.append(word)
    return " ".join(processed_post_text)


def shorten_url(long_url):
    # Use the goo.gl api to shorten a link
    post_url = 'https://www.googleapis.com/urlshortener/v1/url?key=' + \
               credentials.api_key
    params = json.dumps({'longUrl': long_url})
    response = requests.post(post_url,
                             params,
                             headers={'Content-Type': 'application/json'})
    return response.json()['id']


def gen_markov_status(screen_name, timeline, short_url):
    processed_timeline_text = [preprocess_post(post) for post in timeline]
    text_model = markovify.text.NewlineText("\n".join(processed_timeline_text))
    status_len = STATUS_MAX_LEN - (len(screen_name) + 2) - (len(short_url) + 1)
    return "@" + screen_name + " " + \
           text_model.make_short_sentence(status_len) + \
           " " + short_url


def gen_lstm_status(screen_name, timeline, short_url, depth):
    # Create a vector of words and their frequency in on the user's timeline.
    # Experimentation shows that requiring a word to occur at least 4 * depth
    # times to be considered gives good results.
    with open("stopwords.txt", 'r') as stopwords_file:
        stopwords = [line.strip() for line in stopwords_file]
    processed_timeline_text = [preprocess_post(post) for post in timeline]

    vectorizer = CountVectorizer(min_df=4*depth, stop_words=stopwords)
    X = vectorizer.fit_transform(processed_timeline_text)
    vocab = vectorizer.get_feature_names()
    topic = random.choice(vocab)

    # Generates a status using a helper bash script.
    proc = subprocess.Popen([NN_SAMPLE_COMMAND, topic], stdout=subprocess.PIPE)
    status = topic + " " + proc.stdout.read().split("\n")[-2].strip()
    return "@" + screen_name + " " + status + " " + short_url


def post_status_and_sleep(status, depth):
    post_status.post_status(status)
    # Sleep, but optimize: post_status.post_status sleeps as well
    time_to_sleep = SECONDS_PER_TIMELINE * depth - SECONDS_TO_POST
    time.sleep(time_to_sleep)


def schedule_status_and_sleep(status, timeline, depth):
    # Find time to post
    times = []
    for post in timeline:
        time_raw = time.strptime(post['created_at'], CREATED_AT_FORMAT)
        time_formatted = time.strftime('%H', time_raw)
        times.append(time_formatted)

    # collections.Counter.most_common(1) will only return the most common item
    # however, it's wrapped in a tuple inside a list: [(time, frequency)].
    post_time = collections.Counter(times).most_common(1)[0][0]

    # Schedule posting the status
    # The command which calls the script which posts the status
    stripped_status = status.replace("'", "").replace('"', '')
    status_command = "python post_status.py \"" + stripped_status + "\""

    # Wrap the status command using echo, so the status command can
    # be piped to another process
    echo_wrapper = "echo '" + status_command + "'"

    # randomize minute of posted time
    random_minute = random.randint(0,59)
    
    # shell command for scheduling the script for the given time
    at_command = " at " + str(post_time) + ":" + "{:0>2}".format(random_minute)

    # put it all together
    proc = subprocess.call(echo_wrapper + "|" + at_command, shell=True)

    time.sleep(max(SECONDS_PER_TIMELINE * depth, 0))


def get_timeline(depth):
    auth = tweepy.OAuthHandler(credentials.consumer_key,
                               credentials.consumer_secret)
    auth.set_access_token(credentials.access_token,
                          credentials.access_token_secret)
    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

    timeline = []
    for page_num in xrange(depth):
        timeline.extend(api.user_timeline(screen_name=screen_name,
                                          count=MAX_TIMELINE_POSTS,
                                          include_rts=False,
                                          page=page_num))

    return timeline


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Posts statuses to users"
                                                 " given targets and url")
    parser.add_argument("targets",
                        help="textfile list of users to target")
    parser.add_argument("long_url",
                        help="URL which targets should click")
    parser.add_argument("model",
                        help="model to use when generating text",
                        choices=["markov", "lstm"])
    parser.add_argument("depth",
                        help="number of calls to the"
                             " timeline endpoint per user",
                        type=int)

    parser.add_argument("--disable-targeting",
                        help="skip triaging low-value/low-likelihood users",
                        action="store_true")
    parser.add_argument("--disable-scheduling",
                        help="post status immediately instead of scheduling"
                             " for when the user is likely to respond",
                        action="store_true")
    parser.add_argument("--disable-post",
                        help="do not post status; print it to standard out",
                        action="store_true")
    args = parser.parse_args()

    # Read list of potential targets from file
    with open(args.targets, 'r') as targets_file:
        potential_targets = [target.strip() for target in targets_file]

    for screen_name in potential_targets:
        if is_target(screen_name, args.disable_targeting):
            try:
                short_url = shorten_url(args.long_url)
                timeline = get_timeline(args.depth)
                if args.model == "markov":
                    status = gen_markov_status(screen_name,
                                               timeline,
                                               short_url)
                elif args.model == "lstm":
                    status = gen_lstm_status(screen_name,
                                             timeline,
                                             short_url,
                                             args.depth)
                if args.disable_post:
                    print status
                    time.sleep(max(SECONDS_PER_TIMELINE*args.depth, 0))
                elif args.disable_scheduling:
                    post_status_and_sleep(status, args.depth)
                else:
                    schedule_status_and_sleep(status, timeline, args.depth)
            except Exception as e:
                print e
                continue

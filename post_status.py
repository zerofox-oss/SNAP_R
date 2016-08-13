# THIS PROGRAM IS TO BE USED FOR EDUCATIONAL PURPOSES ONLY.
# CAN BE USED FOR INTERNAL PEN-TESTING, STAFF RECRUITMENT, SOCIAL ENGAGEMENT

import credentials
import argparse
import tweepy

# Posts a new status
def post_status(text):
    auth = tweepy.OAuthHandler(credentials.consumer_key,
                               credentials.consumer_secret)
    auth.set_access_token(credentials.access_token,
                          credentials.access_token_secret)
    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
    api.update_status(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Posts status to timeline"
                                                 " of user given"
                                                 " in credentials.py")
    parser.add_argument("text", help="text to post")
    args = parser.parse_args()
    post_status(args.text)

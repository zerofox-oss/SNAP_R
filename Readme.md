# SNAP_R

THIS PROJECT IS TO BE USED FOR EDUCATIONAL PURPOSES ONLY. Obtain explicit consent from users before using this tool.
A project that demonstrates the capability for posting phishing posts to social network users.

# Description

Many toolkits (e.g. SET) provide for automatically generating phishing payloads, but nothing currently exists to automatically generate front-end content to entice users to click on links. Furthermore, social media is an interesting attack surface: users provide a wealth of information which networks readily provide through REST APIs, and due to character limitations, links are generally shortened and grammar tends to be colloquial.

This project demonstrates the capability for automatically generating spear-phishing posts on social media. It uses two methods for generating text for the post: Markov models trained on the target's recent timeline statuses, and an LSTM Neural Network trained on a more general corpus. It also shortens the given url using goo.gl and appends that to the post, prepends the post with their username, and triages users to those with high engagement or value.

# Requirements
* Python 2.7
* Active Twitter developer API credentials, a Twitter account username and password, and a goo.gl API key (all to be placed in the corresponding variables in credentials.py)
* word-rnn, downloaded and installed from github.com/larspars/word-rnn

# To Run:
1. Clone this repository.
2. In the root of the repository, create and fill in credentials.py with your obtained credentials from the various services.
3. Download tweets_model.t7 and move into word-rnn/cv/
4. Obtain a list of users and a URL that you want them to click on.
5. Run pip install -r requirements.txt inside a virtual environment.
6. Run python main.py. The various options and parameters are available if you run python main.py -h.

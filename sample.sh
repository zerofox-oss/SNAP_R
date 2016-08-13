#!/bin/bash
cd ~/word-rnn
th sample.lua cv/tweets_model.t7 -gpuid 0 -primetext "$1" -length 10
cd ~/blackhat

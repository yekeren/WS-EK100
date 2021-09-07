#!/bin/sh

ps x | grep python | grep -v grep | grep -v tensorboard | grep -v jupyter | awk '{print $1}' | xargs kill

#!/bin/bash

# no need to use this if you're cloning the github repo, only if you're distributing the zipped assignment

zip -r reinforcement.zip reinforcement --exclude *.idea* --exclude *.git* --exclude *.pyc -q
echo "If there were no errors, the kit has been created in reinforcement.zip"


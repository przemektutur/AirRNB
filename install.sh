#!/bin/bash

# Script for automagicat installation of Python dependencies
echo "Installation from requirements.txt"
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "Successfull installation!"
else
    echo "Error while installing Python libs (do it manually)."
fi

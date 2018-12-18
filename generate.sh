#!/bin/bash

echo "-> Installing requirements.txt"
pip install -r requirements.txt
echo "-> Building Machine Learning Modules"
cd ml_code
python service.py
cd ..
echo "-> Load migrations"
python manage.py makemigrations
echo "-> Migrating"
python manage.py migrate
echo "-> Running Server"
python manage.py runserver

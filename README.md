# Policies Sparkle Project Backend

This Backend has been written in Flask and has been Deployed on [Render](https://policies-app-backend.onrender.com/).

To run the backend application locally:

First install the `virtualenv` library from PyPI:
```
pip install virtualenv
```
Next, create a virtual environment on python using:

```
python -m venv env
```
or as below, depending on the operating system of the user
```
python3 -m venv env
```

After creating the virtual environment, run the following to activate the virtual environment:

```
source env/bin/activate
```

Upon activating the virtual environment, the user can run the following commands to get the application up and running:

```
pip install -r requirements.txt
python3 app.py #or python app.py 
```

The application should now be running on the localhost with the default port number of 5000.

The deployment version of the application is a WGSI server which uses the `gunicorn` library and it can be run using the command:
```
gunicorn app:app
```
Note that the above command will not work on native windows and requires Windows Subsystem For Linux(WSL) in order to work properly

# AirRNB

## QUICKLY PREPARED PROGRAM FOR SHOW RNB PROJECT 

## FOR LODGIFY ;)

Program predicts a price for one night in various location.

Three cities were added for prediction: London, Athens and Barcelona.

Data is taken from AirRNB.

### SORRY FOR NOT KEEPING OOP AND SPAGHETTI CODE - FAST CODING 

### IN THE MORNING;)

I saw Lodgify is searching for machine learning/deep learning/AI engineer and

as a fast-growing company leading the vacation rental industry with innovative 

software... I thought to prepare in the morning short program which will 

calculate rental prices. I didn't have much time before push kids to the

kindergarten... ;)

so program works, predicts, but for sure code should be cleared and change into

OOP :)

### IF YOU WOULD LIKE TO HAVE IT MADE AS OOP - PLEASE LET ME KNOW

Check 2407-change-into-object-oriented-paradigm branch (I've started to implement

object oriented programming paradigm).


### 1. Check for Python Installation

Ensure you have Python installed on your Linux system. Most Linux distributions

come with Python pre-installed. You can check the installed Python version by 

running:

```bash
python3 --version
```

If Python is not installed or you need a different version, you can typically 

install it using your distribution's package manager. For example, on Ubuntu 

or Debian-based systems, you can install Python 3 using:

```bash
sudo apt update
sudo apt install python3
```

### 2. Install Required Libraries

Application requires several libraries, such as `pandas`, `numpy`, `xgboost`, 

and `tkinter`. You can install these using `pip`, Python's package installer. 

Run the following command to install the required libraries:

```bash
pip3 install pandas numpy xgboost tk
```

Note: `tkinter` might already be installed with Python. If not, you can 

install it via your Linux distribution's package manager. For example, 

on Ubuntu, you can install it with:

```bash
sudo apt-get install python3-tk
```

### 3. Running Application

Navigate to the directory containing application files in the terminal. 

If application entry point is `main.py` (which includes the instantiation of 

your `GUI` class and calls the `run_gui` method), run the application with:

```bash
python3 main.py
```

This command starts your application, and you should see the GUI window 

appear on your screen.

### 4. Interact with Your Application

Now that your application is running, you can test its functionalities:

- Select a city from the dropdown menu.

- Enter the required input values in the text fields.

- Click the "Predict Price" button to see how the application responds,

  including displaying the predicted price and any error messages.

### 4. Optional: Creating a Virtual Environment

For better management of your project's dependencies, consider using a 

virtual environment. This isolates your project's libraries from the global

Python installation. To create and activate a virtual environment, use:

```bash
python3 -m venv myprojectenv
source myprojectenv/bin/activate
```

After activation, any Python or pip commands will use the versions in 

the virtual environment rather than the global ones.


## SCREENSHOTS

XGBRegressor for Barcelona

![XGBRegressor for Barcelona](ML001.png)

XGBRegressor for London

![XGBRegressor for London](ML002.png)

Help

![Help](ML003.png)

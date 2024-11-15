# Rental Price Prediction

## Project Overview

This project aims to predict rental prices for short-term accommodations in

various locations using machine learning. It focuses on three major cities:

London, Athens, and Barcelona, with data sourced from AirBnB. Designed as a

quick morning project before family commitments, it evolved into a 

comprehensive analysis utilizing the XGBoost regression algorithm to forecast

one-night rental prices.


Initially started as a script, the project's codebase underwent significant

refactoring to adopt an object-oriented programming paradigm, facilitating

scalability and maintenance. This transition is documented in the 

'2407-change-into-object-oriented-paradigm' branch, showcasing the iterative 

development process over approximately 5 hours.

## Architecture and Implementation

### Data Preprocessing

Data from AirBnB is preprocessed to prepare it for machine learning models. 

This involves cleaning the data, encoding categorical variables, handling

missing values, and normalizing features to ensure optimal model performance.

### Machine Learning Model

After preprocessing, the project leverages the XGBoost algorithm—a powerful,

efficient, and scalable implementation of gradient boosting—to predict rental

prices. XGBoost was chosen for its performance and ability to handle sparse 

data from AirBnB listings.

### GUI Application

The project features a graphical user interface (GUI) built with Tkinter,

allowing users to select a city, input accommodation features, and predict 

rental prices with a simple click. The GUI makes the application accessible 

to users without a technical background, providing a user-friendly way to 

estimate rental prices.

### Visualization

To offer insights into the data, the application includes a visualization

feature that plots geographical price distributions across the selected cities.

This helps users understand price variations and identify trends related to 

location.

### Object-Oriented Paradigm

The shift to an object-oriented paradigm enhances the project's structure,

making it more modular and easier to extend. This approach encapsulates data

preprocessing, model training, prediction, and visualization functionalities 

within distinct classes, promoting code reuse and maintainability.

## Getting Started

Installation: Instructions on setting up the project environment, including 

required libraries and dependencies.

Usage: How to run the application, including launching the GUI and making 

predictions.

### 1. Check for Python Installation

Ensure you have Python installed on your system. Most Linux distributions

come with Python pre-installed. You can check the installed Python version by 

running in Linux:

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
pip3 install pandas numpy xgboost tk scikit-learn geopy matplotlib 
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

XGBRegressor Linux TK

![XGBRegressor for Linux](pictures/ML004.png)

XGBRegressor for Barcelona

![XGBRegressor for Barcelona](pictures/ML001.png)

XGBRegressor for London

![XGBRegressor for London](pictures/ML002.png)

Help

![Help](pictures/ML003.png)


## FUTURE UPDATES

- add multithreading - I hate slowness of this app (unfortunatelly 

  data is loaded directly from website and it is huge csv for every

  city :/ - data transformation takes some time, but we will handle!) 

- unit tests and test cases for application - oh, no.. seems I will

  be busy as the devil in the hell ;)

- add other Machine Learning algorithms (yeah! - this is what we would

  like to have possibility to choose and check how good each alghoritm

  is for the same city)

## UPDATES

- ugh, returning after some longer vacation time spent on riding motocycle

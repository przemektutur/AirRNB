import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import tkinter.messagebox as messagebox


def preprocess_data(dataL):
    """Data preprocessing function."""
    dataL1 = dataL.isna().sum() > 0
    dataL.loc[:, dataL1].isna().sum().sort_values(
        ascending=False
    ) / len(dataL) * 100
    dataL.drop([
        'description',
        'neighbourhood_group_cleansed',
        'license',
        'calendar_updated',
        'bedrooms',
        'amenities'
    ], axis=1, inplace=True)

    dataL['bathrooms'] = dataL['bathrooms_text'].str.extract(
        '([\d.]+)'
    ).astype(float)
    dataL['bathrooms'] = np.where(
        dataL['bathrooms_text'] == 'Half-bath', 0.5, dataL['bathrooms']
    )
    dataL['bathrooms'].mode()
    dataL['bathrooms'].fillna(1, inplace=True)
    dataL['bathrooms'].unique()
    dataL[['host_is_superhost',
        'host_has_profile_pic',
        'host_identity_verified',
        'has_availability',
        'instant_bookable']].head(3)
    mapping_func = {'t': 1, 'f': 0}
    kolumnytf = ['host_is_superhost',
        'host_has_profile_pic',
        'host_identity_verified',
        'has_availability',
        'instant_bookable']
    dataL[kolumnytf] = dataL[kolumnytf].applymap(
        lambda x: mapping_func.get(x, 0)
    ).astype(bool)
    kolumnypr = ['host_response_rate', 'host_acceptance_rate']
    dataL[kolumnypr] = dataL[kolumnypr].apply(
        lambda x: x.str.replace('%', '').astype(float) / 100
    )
    dataL[kolumnypr].isna().sum() / len(dataL[kolumnypr]) * 100
    dataL['host_response_rate'].fillna(
        round(dataL['host_response_rate'].mean(), 2), inplace=True
    )
    dataL['host_acceptance_rate'].fillna(
        round(dataL['host_acceptance_rate'].mean(), 2), inplace=True
    )
    kolumnydat = [
        'last_scraped',
        'host_since',
        'calendar_last_scraped',
        'first_review',
        'last_review'
    ]
    dataL[kolumnydat] = dataL[kolumnydat].apply(pd.to_datetime)

    dataL['price'] = dataL['price'].replace('\$', '', regex=True).replace(
        ',', '', regex=True
    ).astype(float)
    filtered = dataL[~np.isnan(dataL['price'])]['price']
    perc = np.percentile(filtered, 95)
    dataL = dataL[dataL['price'] < perc]
    dataL.loc[:, dataL1].select_dtypes(
        include='number'
    ).isna().sum().sort_values(ascending=False) / len(dataL) * 100
    dataH = dataL[(dataL['reviews_per_month'] < 1.5)]['reviews_per_month']
    dataH = dataL[(dataL['reviews_per_month'] < 50) &
            (dataL['reviews_per_month'] > 10)]['reviews_per_month']
    filtered = dataL[~np.isnan(
        dataL['reviews_per_month']
    )]['reviews_per_month']
    perc = np.percentile(filtered, 95)
    dataL['reviews_per_month'].isna().sum()
    filtered = filtered[filtered < perc]
    dataL = dataL[dataL['reviews_per_month'] < perc]
    dataL.loc[:, dataL1].select_dtypes(
        include='number'
    ).isna().sum().sort_values(ascending=False) / len(dataL) * 100
    dataL['beds'].fillna(1, inplace=True)
    dataL['beds'].isna().sum()
    rev = [
        'review_scores_value',
        'review_scores_location',
        'review_scores_checkin',
        'review_scores_communication',
        'review_scores_accuracy',
        'review_scores_cleanliness'
    ]
    dataL[rev].agg(['median', 'mean']).T
    for kol in rev:
        dataL[kol].fillna(dataL[kol].mean(), inplace=True)
    dataL[rev].isna().sum()
    label_encoder = LabelEncoder()
    dataL['room_type_encoded'] = label_encoder.fit_transform(
        dataL['room_type']
    )
    dataL['neighbourhood_cleansed_encoded'] = label_encoder.fit_transform(
        dataL['neighbourhood_cleansed']
    )
    all_num_feats = [x for x in dataL.dtypes[(dataL.dtypes != 'object') &
        (dataL.dtypes != 'datetime64[ns]')].index
        if not ('id' in x) and not ('price' in x)]
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    dataL = dataL[dataL['maximum_nights'] <= np.percentile(
        dataL['maximum_nights'],
        95
    )]
    best_featuresLX = [
        'accommodates',
        'room_type_encoded',
        'latitude',
        'longitude',
        'bathrooms',
        'review_scores_rating',
        'availability_365',
        'has_availability',
        'beds',
        'neighbourhood_cleansed_encoded',
        'availability_30',
        'review_scores_cleanliness',
        'reviews_per_month',
        'calculated_host_listings_count_entire_homes',
        'number_of_reviews_ltm',
        'availability_90'
    ]

    dataL_subset = dataL.loc[:, best_featuresLX + ['price']]
    return dataL_subset

def train_model(X, y):
    model = xgboost.XGBRegressor(n_estimators=200, verbosity=1)
    model.fit(X, y)
    return model

def on_predict_button_click():
    if selected_data_var.get() == "Barcelona":
        selected_data = dataB
    elif selected_data_var.get() == "London":
        selected_data = dataL
    elif selected_data_var.get() == "Athens":
        selected_data = dataT

    address = input_entries['address'].get()

    # Geopy coordinates
    coordinates = get_coordinates(address)
    input_values = {}
    lat = ''
    long = ''
    if coordinates:
        input_values = {
            'latitude': coordinates[0],
            'longitude': coordinates[1]
        }
        lat = coordinates[0]
        long = coordinates[1]
    else:
        print("Problem with getting values.")
        return

    for feature in selected_features:
        if feature not in ['latitude', 'longitude']:
            input_values[feature] = float(input_entries[feature].get())

    for feature in mean_features:
        input_values[feature] = selected_data[feature].mean()

    input_df = pd.DataFrame([input_values])

    best_features = [
        'latitude',
        'longitude',
        'accommodates',
        'room_type_encoded',
        'bathrooms',
        'review_scores_rating',
        'availability_365',
        'has_availability',
        'beds',
        'neighbourhood_cleansed_encoded',
        'availability_30',
        'review_scores_cleanliness',
        'reviews_per_month',
        'calculated_host_listings_count_entire_homes',
        'number_of_reviews_ltm',
        'availability_90'
    ]

    data_subset = selected_data.loc[:, best_features + ['price']]
    X = data_subset.loc[:, best_features]
    y = np.array(data_subset['price'])

    model = train_model(X, y)
    predicted_price = model.predict(input_df)[0]
    text = (
        f'Predicted Price: ${predicted_price:.2f}\n'
        f'Latitude: {lat}\nLongitude:{long}'
    )
    prediction_label.config(text=text)

def data_plot(data0, data1, data2, title0, title1, title2):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    scatter1 = ax1.scatter(
        data1['longitude'],
        data1['latitude'],
        c=data1['price'],
        cmap='jet',
        vmax=200,
        vmin=25
    )
    ax1.set_title(title1)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Price')

    scatter2 = ax2.scatter(
        data2['longitude'],
        data2['latitude'],
        c=data2['price'],
        cmap='jet',
        vmax=200,
        vmin=25
    )
    ax2.set_title(title2)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Price')

    scatter3 = ax3.scatter(
        data0['longitude'],
        data0['latitude'],
        c=data0['price'],
        cmap='jet',
        vmax=200,
        vmin=25
    )
    ax3.set_title(title0)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Price')

    plt.tight_layout()
    plt.show()

def get_coordinates(address):
    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.geocode(f"{address}")
    if location:
        return location.latitude, location.longitude
    else:
        return None

def show_about():
    legend = (
        "Room type encoded:\n"
        "3 - shared room\n"
        "2 - private room\n"
        "1 - hotel room\n"
        "0 - entire home/apt"
    )
    messagebox.showinfo("Legend: ", legend)

def show_team():
    messagebox.showinfo("Team: Przemyslaw Tutur")

prefix = "http://data.insideairbnb.com/"
# Load data for Barcelona
dataB = pd.read_csv(
    prefix + "spain/catalonia/barcelona/2023-12-13/data/listings.csv.gz"
)
# Load data for London
dataL = pd.read_csv(
    prefix + "united-kingdom/england/london/2023-12-10/data/listings.csv.gz"
)
# Load data for Athens
dataT = pd.read_csv(
    prefix + "greece/attica/athens/2023-12-25/data/listings.csv.gz"
)
# Choose the default dataset
selected_data = dataL
# GUI setup
root = tk.Tk()
root.title("Predictor")
root.geometry("300x450")

help_menu = tk.Menu(root)
root.config(menu=help_menu)

# Create the 'Legend' submenu
legend_submenu = tk.Menu(help_menu, tearoff=0)
help_menu.add_cascade(label="Legend", menu=legend_submenu)
legend_submenu.add_command(label="Help", command=show_about)

# Create the 'Team' submenu
team_submenu = tk.Menu(help_menu, tearoff=0)
help_menu.add_cascade(label="Team", menu=team_submenu)
team_submenu.add_command(label="About", command=show_team)

# Dropdown menu for selecting dataset
selected_data_var = tk.StringVar(root)
selected_data_var.set("Barcelona")  # Default selection

data_dropdown_label = tk.Label(root, text="Select City:")
data_dropdown_label.pack(pady=5)

data_dropdown = ttk.Combobox(
    root,
    textvariable=selected_data_var,
    values=["Barcelona","London", "Athens"]
)
data_dropdown.pack(pady=5)

# Preprocess the data
dataB = preprocess_data(dataB)
dataL = preprocess_data(dataL)
dataT = preprocess_data(dataT)

# Extract features and target variable
best_featuresLX = [
    'accommodates',
    'room_type_encoded',
    'latitude',
    'longitude',
    'bathrooms',
    'review_scores_rating',
    'availability_365',
    'has_availability',
    'beds',
    'neighbourhood_cleansed_encoded',
    'availability_30',
    'review_scores_cleanliness',
    'reviews_per_month',
    'calculated_host_listings_count_entire_homes',
    'number_of_reviews_ltm',
    'availability_90'
]

dataL_subset = dataL.loc[:, best_featuresLX + ['price']]
X = dataL_subset.loc[:, best_featuresLX]
y = np.array(dataL_subset['price'])
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Train the model
model = train_model(X_train, y_train)

# Labels and Entry Widgets for Selected Features
selected_features = ['accommodates', 'room_type_encoded', 'bathrooms']
input_entries = {}

for feature in selected_features:
    label = tk.Label(root, text=feature.capitalize() + ":")
    label.pack(pady=5)
    entry = tk.Entry(root)
    entry.pack(pady=5)
    input_entries[feature] = entry

label = tk.Label(root, text="Address" + ":")
label.pack(pady=5)
entry = tk.Entry(root)
entry.pack(pady=5)
input_entries["address"] = entry

# Labels for Features with Mean Values
mean_features = [
    'review_scores_rating',
    'availability_365',
    'has_availability',
    'beds',
    'neighbourhood_cleansed_encoded',
    'availability_30',
    'review_scores_cleanliness',
    'reviews_per_month',
    'calculated_host_listings_count_entire_homes',
    'number_of_reviews_ltm',
    'availability_90'
]

for feature in mean_features:
    mean_value = dataL[feature].mean()

# Prediction Button
predict_button = tk.Button(
    root,
    text="Predict Price",
    command=on_predict_button_click
)
predict_button.pack(pady=10)

# Prediction Result Label
prediction_label = tk.Label(root, text="")
prediction_label.pack(pady=10)

data_plot(dataB, dataL, dataT, 'Barcelona Data', 'London Data', 'Athens Data')
root.mainloop()

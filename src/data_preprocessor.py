"""Data Preprocessror class."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """DataPreprocessor class."""

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initializes the DataPreprocessor with the dataset to be processed.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to preprocess.
        """
        self.data: pd.DataFrame = data

    def preprocess(self) -> None:
        """
        Performs preprocessing on the dataset to clean and prepare data 
        for further analysis or modeling.

        This includes:
        - Dropping unnecessary columns
        - Extracting numeric data from text
        - Filling missing values
        - Encoding categorical variables
        """
        dataL: pd.DataFrame = self.data
        dataL.drop(
            [
                'description', 
                'neighbourhood_group_cleansed', 
                'license', 
                'calendar_updated', 
                'bedrooms', 
                'amenities'
            ], 
            axis=1, 
            inplace=True
        )
        dataL['bathrooms'] = dataL['bathrooms_text'].str.extract(
                '([\d.]+)'
            ).astype(float)
        dataL['bathrooms'] = np.where(
            dataL['bathrooms_text'] == 'Half-bath', 0.5, dataL['bathrooms']
        )
        dataL['bathrooms'].fillna(dataL['bathrooms'].mode()[0], inplace=True)

        boolean_columns = [
            'host_is_superhost', 
            'host_has_profile_pic', 
            'host_identity_verified', 
            'has_availability', 
            'instant_bookable'
        ]
        for col in boolean_columns:
            dataL[col] = dataL[col].map({'t': True, 'f': False})

        percent_columns = ['host_response_rate', 'host_acceptance_rate']
        for col in percent_columns:
            dataL[col] = dataL[col].str.rstrip('%').astype('float') / 100.0

        date_columns = [
            'last_scraped', 
            'host_since', 
            'calendar_last_scraped', 
            'first_review', 
            'last_review'
        ]
        for col in date_columns:
            dataL[col] = pd.to_datetime(dataL[col])

        dataL['price'] = dataL['price'].replace(
            {'\$': '', ',': ''}, 
            regex=True
        ).astype(float)

        le = LabelEncoder()
        dataL['room_type_encoded'] = le.fit_transform(dataL['room_type'])
        dataL['neighbourhood_cleansed_encoded'] = le.fit_transform(
            dataL['neighbourhood_cleansed']
        )
        self.data = dataL

    def get_data(self) -> pd.DataFrame:
        """
        Returns the preprocessed dataset.

        Returns
        -------
        pd.DataFrame
            The preprocessed dataset ready for analysis or modeling.
        """
        return self.data


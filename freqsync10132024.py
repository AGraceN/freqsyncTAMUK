import numpy as np
import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sqlalchemy import create_engine, text, Column, Integer, Float, Table, MetaData, inspect
from sqlalchemy.orm import declarative_base

initialTrends = []
GeneratingUnitInertia = {'nuclear': 5.9, 'gas': 4.2, 'coal': 4.2, 'hydro': 2.4, 'solar': 0.01, 'storage': 0.01, 'wind': 0.01}
GeneratingUnitTypeCount = {'nuclear': 4, 'gas': 44, 'coal': 10, 'hydro': 1, 'solar': 13, 'storage': 3, 'wind': 25}
eventMagnitude = {'small': 0.00125, 'medium': 0.0025, 'large': 0.00375, 'serious': 0.005}
generatingUnitsCount = 100
generatingUnitColumnNames = [f'GeneratingUnit{i+1}' for i in range(generatingUnitsCount)]
connection_string = "mysql+mysqlconnector://root:password@localhost:3306/" # Replace Username, Password, FQDN, and Port with local MySql server connection information
db_name = "freqsync"

plt.rcParams['figure.max_open_warning'] = 0

def database_exists(engine, db_name):
    with engine.connect() as conn:
        result = conn.execute(text(f"SHOW DATABASES LIKE '{db_name}'"))
        return result.fetchone() is not None

def create_database(engine, db_name):
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
        conn.commit()

def drop_database(engine, db_name):
    with engine.connect() as conn:
        conn.execute(text(f"DROP DATABASE {db_name}"))
        conn.commit()

db_engine = create_engine("mysql+mysqldb://root:password@localhost")  # Replace Username, Password, and FQDN with local MySql server connection information
if database_exists(db_engine, db_name):
    drop_database(db_engine, db_name)
create_database(db_engine, db_name)
db_engine.dispose()
engine = create_engine(f"{connection_string}{db_name}")

inertia_metadata = MetaData()
inertia_columns = [Column(f'GeneratingUnit{j+1}', Float) for j in range(generatingUnitsCount)]
inertia_table = Table('unitinertia', inertia_metadata, *inertia_columns)
inertia_metadata.create_all(engine)

inertia_data = {}
unit_index = 0
for unit_type, count in GeneratingUnitTypeCount.items():
    for _ in range(count):
        unit_name = f'GeneratingUnit{unit_index + 1}'
        inertia_data[unit_name] = GeneratingUnitInertia[unit_type]
        unit_index += 1

pd.DataFrame([inertia_data]).to_sql('unitinertia', engine, if_exists='append', index=False)
    
for magnitude_name, event in eventMagnitude.items():

    metadata = MetaData()
    if inspect(engine).has_table(f'initialtrends_{magnitude_name}'):
        Table(f'initialtrends_{magnitude_name}', metadata, autoload_with=engine).drop(engine)

    trends_metadata = MetaData()
    trends_columns = [Column('Date', Integer)] + [Column(f'GeneratingUnit{i+1}', Float) for i in range(generatingUnitsCount)]
    trends_table = Table(f'initialtrends_{magnitude_name}', trends_metadata, *trends_columns)
    trends_metadata.create_all(engine)



    sensorValues = {col: [] for col in generatingUnitColumnNames}
    generationUnitSequenceCount = -1

    for key, count in GeneratingUnitTypeCount.items():
        inertia = GeneratingUnitInertia[key]
        for _ in range(count):
            generationUnitSequenceCount += 1
            column_name = generatingUnitColumnNames[generationUnitSequenceCount]
            sensorValue = 60.0
            frequencyChangePerSecond = (event / inertia) / 100

            unit_values = [sensorValue - frequencyChangePerSecond * b for b in range(100)]
            sensorValues[column_name] = unit_values

    with engine.connect() as connection:
        for row in range(100):
            values = {"date": row + 1}
            values.update({col: sensorValues[col][row] for col in generatingUnitColumnNames})

            insert_query = text(
                f"INSERT INTO initialtrends_{magnitude_name} (Date, {', '.join(generatingUnitColumnNames)}) "
                f"VALUES (:date, {', '.join([f':{col}' for col in generatingUnitColumnNames])})"
            )
            connection.execute(insert_query, values)
        connection.commit()

    forecast_dict = {}
    observed_df = None

    for freq in range(1, 101):
        query = f"SELECT GeneratingUnit{freq} FROM initialtrends_{magnitude_name} WHERE GeneratingUnit{freq} IS NOT NULL"
        df = pd.read_sql_query(query, engine)

        if df.empty:
            continue

        model = UnobservedComponents(df[f"GeneratingUnit{freq}"], 'local linear trend')
        result = model.fit(disp=False, method='powell')

        with open(f"GeneratingUnit{freq}_{magnitude_name}_StateSpaceModelResults.txt", 'w') as f:
            f.write(str(result.summary()))

        predict = result.get_prediction()
        forecast = result.get_forecast(steps=10)

        fig, ax = plt.subplots(figsize=(10, 4))
        df[f"GeneratingUnit{freq}"].plot(ax=ax, style='k.', label='Observations')
        predict.predicted_mean.plot(ax=ax, label='One-step-ahead Prediction')

        forecast.predicted_mean.plot(ax=ax, style='r', label='Forecast')
        ax.legend(loc='lower left')
        plt.savefig(f"GeneratingUnit{freq}_{magnitude_name}.png")
        plt.close()

        forecast_dict[f'GeneratingUnit{freq}'] = forecast.predicted_mean

    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df.to_sql(f'forecast_output_{magnitude_name}', engine, if_exists='replace', index_label='date')

    inertia = pd.read_sql_query(text("""SELECT * FROM unitinertia"""), engine)
    frequency = pd.read_sql_query(text(f"SELECT * FROM forecast_output_{magnitude_name} WHERE date IN (SELECT MAX(date) FROM forecast_output_{magnitude_name})"), engine)

    frequency.drop(columns='date', axis=1, inplace=True)
    inertia = inertia.values.ravel()
    frequency = frequency.values.ravel()

    scaler = StandardScaler()
    inertia_scaled = scaler.fit_transform(inertia.reshape(-1, 1))

    adaline_model = make_pipeline(StandardScaler(), SGDRegressor(eta0=0.01, max_iter=1000, tol=1e-3))
    adaline_model.fit(inertia_scaled, frequency)

    optimal_frequency = adaline_model.predict([[inertia_scaled.mean()]])

    f = open( f'optimal_frequency_{magnitude_name}.txt', 'w' )
    f.write(str(optimal_frequency))
    f.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(inertia, frequency, color='blue', label='Original Data')
    plt.axhline(y=optimal_frequency[0], color='red', linestyle='--', label='Optimal Frequency')
    plt.xlabel('Inertia')
    plt.ylabel('Frequency')
    plt.title('Original Data with Predicted Synchronized Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"optimumControlFrequency_{magnitude_name}")

    query3 = f"SELECT * FROM initialtrends_{magnitude_name};"
    golow_df = pd.read_sql_query(query3, engine).set_index('Date')
    golow_df.head()
    golowvalueseries = golow_df.min()
    golowvalue = float(golowvalueseries.min())
    optimalvalueseries = optimal_frequency.max()
    optimalvalue = float(optimalvalueseries)
    golowdiff = 60.0 - golowvalue
    optimaldiff = 60.0 - optimalvalue
    efficiencygainraw = optimaldiff / golowdiff
    efficiencygainpercent = 100 - (efficiencygainraw * 100)
    f = open(f'efficiency_gain_percent_{magnitude_name}.txt', 'w' )
    f.write(str(efficiencygainpercent))
    f.close()

engine.dispose()
print("Processing Completed. All Files Have Been Successfully Written.")

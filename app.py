from fastapi import FastAPI, status
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv(find_dotenv())
connection_uri = os.environ.get("mongodb_uri")
client = MongoClient(connection_uri)
collection = client.healthcare.patients
onehot_encoder = OneHotEncoder()
label_encoder = LabelEncoder()


def setup_db():
    df = pd.read_csv('drugsComTrain_raw.csv')
    df1 = pd.read_csv('drugsComTest_raw.csv')
    df3 = pd.concat([df, df1])
    df3 = df3.dropna()
    df3['age'] = np.random.randint(23, 65, df3.shape[0])
    df3['gender'] = np.random.choice(['male', 'female'], df3.shape[0])
    df3['region'] = np.random.choice(['Africa', 'Asia', 'Caribbean', 'Central America',
                                     'Europe', 'North America', 'Oceania', 'South America'], df3.shape[0])
    df3['year'] = pd.DatetimeIndex(df3['date']).year
    df3['recovery_rate'] = df3['rating']*10
    df3 = df3[df3["condition"].str.contains("</span>") == False]
    df3 = df3.drop_duplicates()
    records = df3.to_dict('records')
    record_list = list(records)
    for i in range(0, len(record_list), 25000):
        client.healthcare.patients.insert_many(record_list[i:i+25000])
    print("Database Setup Successfully Completed")


if os.environ.get("setup") == "True":
    setup_db()


def model_train():
    print("running train model")
    df = pd.DataFrame(list(collection.find()))
    df.dropna()
    df1 = df.drop(
        ["_id", "uniqueID", "review", "rating", "date", "usefulCount"], axis=1
    )
    data = df1.drop_duplicates().reset_index(drop=True)
    X = data.loc[:, ["condition", "age", "recovery_rate"]]
    y = data.loc[:, "drugName"]
    numeric_features = ["age", "recovery_rate"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, ["condition"]),
        ], remainder="passthrough"
    )
    global model
    model = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("classifier", DecisionTreeClassifier())]
    )
    model.fit(X, y)


model_train()


class train_body(BaseModel):
    condition: str
    age: int
    recovery_rate: Optional[int] = 50


class insert_record(BaseModel):
    uniqueID: int
    drugName: str
    condition: str
    rating: int
    age: int
    gender: str
    region: str
    date: str


@app.get("/")
def root():
    return {"message": "Hello World!"}


@app.get("/conditions")
def read_course():
    return collection.distinct("condition")


@app.get("/drug-items")
def read_item(drug: str):
    elements = collection.find({"drugName": drug}, projection={
        '_id': False, 'age': True, 'gender': True, 'region': True, 'recovery_rate': True})
    return list(elements)


@app.get("/generate-id")
def read_item():
    elements = collection.count_documents({})
    return elements+1


@app.get("/year-items")
def read_item(year: int):
    elements = list(collection.find({"year": year}, projection={'_id': False}))
    return elements


@app.post("/predict")
def predict(data: train_body):
    averages = list(collection.aggregate([{
        "$group":
        {
            "_id": "$condition",
            "avg_recovery_rate": {"$avg": "$recovery_rate"}
        }}
    ]))
    recovery_rate = int(list(filter(lambda averages: averages['_id'] == data.condition, averages))[
        0]['avg_recovery_rate'])
    result = model.predict(pd.DataFrame(
        dict({'condition': data.condition, 'age': data.age, 'recovery_rate': recovery_rate}), index=[0]))
    return result[0]


@app.post("/create-record", status_code=status.HTTP_201_CREATED)
def predict(data: insert_record):
    year = data.date.split("/")[-1]
    recovery_rate = data.rating*10
    collection.insert_one({"uniqueID": data.uniqueID, "year": year, "recovery_rate": recovery_rate, "drugName": data.drugName,
                          "condition": data.condition, "rating": data.rating, "age": data.age, "gender": data.gender, "region": data.region})
    model_train()
    return {"result": "Record Added Successfully"}

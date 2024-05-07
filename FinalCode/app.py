from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('Obesity-Predictor\model.pkl','rb'))

# required conversion for the model

genderDict = {
    "female": 0,
    "male": 1
}

familyHistDict = {
    "no": 0,
    "yes": 1
}

highCalDict= {
    "no": 0,
    "yes": 1
}

foodBetweenDict= {
    "always": 0,
    "frequently": 1,
    "no": 2,
    "sometimes": 3,
}
smokeDict ={
    "no": 0,
    "yes": 1
}

monitorDict = {
    "no": 0,
    "yes": 1
}

alcoholDict= {
    "always": 0,
    "frequently": 1,
    "no": 2,
    "sometimes": 3,
}

transportDict = {
    "automobile": 0,
    "bike": 1,
    "motorbike": 2,
    "public-transport": 3,
    "walking": 4,

}

predictionDict = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Obesity_Type_I',
    3: 'Obesity_Type_II',
    4: 'Obesity_Type_III',
    5: 'Overweight_Level_I',
    6: 'Overweight_Level_II'
}



@app.route("/")   # used to define the routes on the web pages
def index_page():
    return render_template("index.html")

# @app.route("/mainForm", methods=["GET","POST"])
# def mainForm():
#     return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def predictPage():

    featureList = []
    gender = request.form["gender"] #1
    gender = int(genderDict[gender]) # convert to numerical value
    featureList.append(gender)

    age = int(request.form["age"]) #2
    featureList.append(age)
    
    height = int(float(request.form["height"])) #3
    featureList.append(height)

    weight = int(request.form["weight"]) #4
    featureList.append(weight)


    familyHistory = request.form["family-history"] #5
    familyHistory= int(familyHistDict[familyHistory]) # convert to numerical value
    featureList.append(familyHistory)


    highCalorieFood = request.form["high-calorie-food"] #6
    highCalorieFood= int(highCalDict[highCalorieFood]) # convert to numerical value
    featureList.append(highCalorieFood)


    vegetables = int(request.form["vegetables"]) #7
    featureList.append(vegetables)


    mainMeals = int(request.form["main-meals"]) #8
    featureList.append(mainMeals)


    foodBetweenMeals = request.form["food-between-meals"] #9
    foodBetweenMeals = int(foodBetweenDict[foodBetweenMeals]) # convert to numerical value
    featureList.append(foodBetweenMeals)


    smoking =request.form["smoking"] #10
    smoking = int(smokeDict[smoking]) #convert to numerical value
    featureList.append(smoking)


    waterIntake = int(request.form["water-intake"]) #11
    featureList.append(waterIntake)


    calorieIntakeMonitor = request.form["calorie-intake-monitor"] #12
    calorieIntakeMonitor = int(monitorDict[calorieIntakeMonitor]) # convert to numerical value
    featureList.append(calorieIntakeMonitor)


    physicalActivity = int(request.form["physical-activity"]) #13
    featureList.append(physicalActivity)


    technologyUsage = float(request.form["technology-usage"]) #14
    featureList.append(technologyUsage)


    alcohol = request.form["alcohol"] #15
    alcohol = int(alcoholDict[alcohol]) # convert to numerical value
    featureList.append(alcohol)


    transport = request.form["transport"] #16
    transport = int(transportDict[transport]) # convert to numerical value
    featureList.append(transport)

    
    finalFeatures= np.array([featureList])
    predictedClass = model.predict(finalFeatures)
    
    predictedFinalClass = predictionDict[predictedClass[0]]

    return render_template("index.html", prediction = predictedFinalClass)



if __name__ == "__main__":  
    app.run(debug=True)  # by writing true it automatically detects the changes



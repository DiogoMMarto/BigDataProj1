import graphlib
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from sklearn import tree
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB



data = pd.read_csv('winequality-white.csv', sep=';')

X = data[['pH', 'alcohol', 'residual sugar']]  # Replace with actual column names
Y = data['quality']  # Quality column

def extract_key_values(x):
    MIN = min(x)
    MAX = max(x)
    
    a = 1.5  # Scaling factors
    b = 0.5
    c = 0.5
    
    q1 = np.quantile(x, 0.25)
    q2 = np.quantile(x, 0.50)  # Median
    q3 = np.quantile(x, 0.75)
    
    LL = max(MIN, q1 - a * (q3 - q1))
    HL = min(MAX, q3 + a * (q3 - q1))
    
    # Define v1 and v2 adjustments
    v1 = c * (LL - MIN)
    v2 = c * (MAX - HL)
    
    # Define triangular membership functions (three points each)
    ol = [MIN, MIN, LL]         # Left-most set (ol)
    low = [MIN, q1, q2]         # Low fuzzy set
    normal = [q1, q2, q3]       # Normal fuzzy set
    high = [q2, q3, HL]         # High fuzzy set
    oh = [HL, MAX, MAX]         # Right-most set (oh)
    
    return ol, low, normal, high, oh, MIN, MAX

ol_pH, low_pH, normal_pH, high_pH, oh_pH, pH_min, pH_max = extract_key_values(X.iloc[:, 0].values)
pH = ctrl.Antecedent(np.arange(pH_min, pH_max, 0.01), 'pH')

# Define triangular membership functions for pH
pH['ol'] = fuzz.trimf(np.arange(pH_min, pH_max, 0.01), ol_pH)
pH['low'] = fuzz.trimf(np.arange(pH_min, pH_max, 0.01), low_pH)
pH['normal'] = fuzz.trimf(np.arange(pH_min, pH_max, 0.01), normal_pH)
pH['high'] = fuzz.trimf(np.arange(pH_min, pH_max, 0.01), high_pH)
pH['oh'] = fuzz.trimf(np.arange(pH_min, pH_max, 0.01), oh_pH)

ol_alcohol, low_alcohol, normal_alcohol, high_alcohol, oh_alcohol, alcohol_min, alcohol_max = extract_key_values(X.iloc[:, 1])
alcohol = ctrl.Antecedent(np.arange(alcohol_min, alcohol_max, 0.01), 'alcohol')

alcohol['ol'] = fuzz.trimf(np.arange(alcohol_min, alcohol_max, 0.01), ol_alcohol)
alcohol['low'] = fuzz.trimf(np.arange(alcohol_min, alcohol_max, 0.01), low_alcohol)
alcohol['normal'] = fuzz.trimf(np.arange(alcohol_min, alcohol_max, 0.01), normal_alcohol)
alcohol['high'] = fuzz.trimf(np.arange(alcohol_min, alcohol_max, 0.01), high_alcohol)
alcohol['oh'] = fuzz.trimf(np.arange(alcohol_min, alcohol_max, 0.01), oh_alcohol)

ol_sugar, low_sugar, normal_sugar, high_sugar, oh_sugar, sugar_min, sugar_max = extract_key_values(X.iloc[:, 2])
sugar = ctrl.Antecedent(np.arange(sugar_min, sugar_max, 0.1), 'residual_sugar')

sugar['ol'] = fuzz.trimf(np.arange(sugar_min, sugar_max, 0.1), ol_sugar)
sugar['low'] = fuzz.trimf(np.arange(sugar_min, sugar_max, 0.1), low_sugar)
sugar['normal'] = fuzz.trimf(np.arange(sugar_min, sugar_max, 0.1), normal_sugar)
sugar['high'] = fuzz.trimf(np.arange(sugar_min, sugar_max, 0.1), high_sugar)
sugar['oh'] = fuzz.trimf(np.arange(sugar_min, sugar_max, 0.1), oh_sugar)

quality = ctrl.Consequent(np.arange(3, 9.1, 0.1), 'quality')  # Wine quality from 0 to 10

min_val = min(Y)
max_val = max(Y)
center = 6

quality['low'] = fuzz.trimf(quality.universe, [min_val, min_val, center])      
quality['normal'] = fuzz.trimf(quality.universe, [min_val, center, max_val])   
quality['high'] = fuzz.trimf(quality.universe, [center, max_val, max_val]) 

rule1 = ctrl.Rule(pH['ol'] | (pH['low'] & alcohol['ol']), quality['low'])
rule2 = ctrl.Rule((pH['low'] & alcohol['low']) | (sugar['ol']), quality['low'])
rule3 = ctrl.Rule(pH['normal'] & alcohol['normal'], quality['normal'])
rule4 = ctrl.Rule((pH['high'] | sugar['normal']) & alcohol['normal'], quality['normal'])
rule5 = ctrl.Rule((pH['high'] & alcohol['high']) | sugar['high'], quality['high'])
rule6 = ctrl.Rule(pH['oh'] | alcohol['oh'] | sugar['oh'], quality['high'])

problem_ctrl  = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
problem = ctrl.ControlSystemSimulation(problem_ctrl)


Y_pred = np.zeros(len(X))
res_file = "res.csv"    
N = 3
with open(res_file, 'w') as f:
    for i in range(len(X)):
        print(f"\r{i}", end="")
        try:
            problem.input['pH'] = X.iloc[i, 0]
            problem.input['alcohol'] = X.iloc[i, 1]
            problem.input['residual_sugar'] = X.iloc[i, 2]
            problem.compute()
            Y_pred[i] = problem.output['quality']
            # extract the probblem output to array
            
            low_activation = fuzz.interp_membership(quality.universe, quality['low'].mf, problem.output['quality'])
            normal_activation = fuzz.interp_membership(quality.universe, quality['normal'].mf, problem.output['quality'])
            high_activation = fuzz.interp_membership(quality.universe, quality['high'].mf, problem.output['quality'])

            low_activation_mf = np.clip(quality['low'].mf,None,low_activation)
            normal_activation_mf = np.clip(quality['normal'].mf,None,normal_activation)
            high_activation_mf = np.clip(quality['high'].mf,None,high_activation)

            area_res = np.fmax(low_activation_mf,np.fmax(normal_activation_mf,high_activation_mf))

            # split area_res in N parts 
            area_res_split = np.array_split(area_res,N)
            quality_universe_split = np.array_split(quality.universe,N)
            # find area and centroid of each area_res_split 
            res = np.zeros(N*2)
            for j in range(N):
                area = np.sum(area_res_split[j])
                centroid = np.sum(area_res_split[j] * quality_universe_split[j]) / area if area > 0 else -1
                res[2*j] = area
                res[2*j+1] = centroid
            
            for j in range(N*2-1):
                f.write(f"{res[j]},")
            f.write(f"{res[-1]},{Y[i]}\n")
            
        except Exception as e:
            # print(f"Error processing row {i}: \x1b[31m{e}\x1b[0m")
            Y_pred[i] = 0 
print()
# open res.csv with pandas
res = pd.read_csv(res_file, sep=',')

# extract X2 and Y2 from res.csv
X2 = res.iloc[:, 0:N*2]
Y2 = res.iloc[:, N*2]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)

def polynomial_features(X, degree):
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(X)

P = 2

X_train_P = polynomial_features(X_train, P)
print(X_train_P.shape)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train_P, y_train)

X_test_P = polynomial_features(X_test, P)
# Make predictions on the test set
y_pred2 = np.round(clf.predict(X_test_P))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred2)
print(f"Accuracy DT: {accuracy:.2f}")

acc2 = accuracy_score(Y2, np.round(Y_pred[ Y_pred != 0 ][:-1]))
print(f"Accuracy base: {acc2:.2f}")

real_dict = { i:0 for i in range(10)}
base_dict = { i:0 for i in range(10)}
dt_dict = { i:0 for i in range(10)}

y_pred2 = np.round(clf.predict(polynomial_features(X2, P)))
for i in range(len(Y2)):
    real_dict[int(Y2[i])] += 1
    base_dict[int(np.round(Y_pred[ Y_pred != 0 ][i]))] += 1
    dt_dict[int(y_pred2[i])] += 1

print(real_dict)
print(base_dict)
print(dt_dict)

# plot the results
fig, ax = plt.subplots(4, 1, figsize=(10, 10))

ax[0].bar(real_dict.keys(), real_dict.values())
ax[1].bar(base_dict.keys(), base_dict.values())
ax[2].bar(dt_dict.keys(), dt_dict.values())

plt.show()
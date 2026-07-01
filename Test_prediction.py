import pandas as pd
import joblib

print("Loading the trained model...")

model=joblib.load('random_forest_SOC_model.pkl')
scaler=joblib.load('scaler.pkl')

#Standard White card values are (R=255, G=255, B=255) but in a real case they can be different. So, we will take the input values from the user.
#RGB values for white card can be used to normalize the input RGB values. For example, if the white card has values (R=240, G=240, B=240), we can normalize the input RGB values by dividing them by these white card values.

White_R=int(input("Enter the R value of the white card (0-255): "))
White_G=int(input("Enter the G value of the white card (0-255): "))   
White_B=int(input("Enter the B value of the white card (0-255): "))

R_val=int(input("Enter the R value (0-255): "))
G_val=int(input("Enter the G value (0-255): ")) 
B_val=int(input("Enter the B value (0-255): "))

# Normalize the input RGB values using the white card values
R_normalized = R_val / White_R
G_normalized = G_val / White_G
B_normalized = B_val / White_B

test_data=pd.DataFrame({'R':[R_normalized],'G':[G_normalized],'B':[B_normalized]})
test_data['Brightness'] = (test_data['R'] + test_data['G'] + test_data['B']) / 3.0
test_data = test_data[['R', 'G', 'B', 'Brightness']]

test_data_scaled = scaler.transform(test_data)
predicted_soc = model.predict(test_data_scaled)

print(f"Input RGB: [{R_val}, {G_val}, {B_val}]")
print(f"Normalized RGB: [{R_normalized:.4f}, {G_normalized:.4f}, {B_normalized:.4f}]")
print(f"Predicted Soil Organic Carbon (SOC): {predicted_soc[0]:.2f}%")

import requests
body = {
    "Item_Weight": 12.9,
    "Item_Visibility": 0.023721223,
    "Item_MRP": 32.2432,
    "Outlet_Establishment_Year": 1997
    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())

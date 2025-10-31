import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="customer-conversion-prediction")

# Load your pipeline
#with open('pipeline_v1.bin', 'rb') as f_in:
#    pipeline = pickle.load(f_in)

with open('/code/pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Define input schema
class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(customer: Customer):
    # Convert Pydantic model to dict
    customer_dict = customer.dict()
    
    # You may need to wrap in a list for scikit-learn
    result = pipeline.predict_proba([customer_dict])[0, 1]
    
    print(f'Probability of converting: {result:.3f}')
    return {"conversion_probability": float(result)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

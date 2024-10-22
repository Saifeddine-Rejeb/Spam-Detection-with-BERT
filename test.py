import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch.nn.functional as F  

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load('spam_model.pt'))
model.eval()  


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = F.softmax(logits, dim=-1).numpy()
    prediction = np.argmax(probabilities, axis=-1)
    return prediction, probabilities

if __name__ == "__main__":
    test_text = """
                    Dear Ms. Johnson,

                    I hope this email finds you well. I am writing to formally request a meeting to discuss the upcoming marketing strategy for the next quarter. During this meeting, we will review the current progress, analyze key performance indicators, and propose new initiatives to improve our outreach and engagement.

                    Could you kindly confirm your availability for a meeting next week? I am available on Tuesday and Thursday between 10 AM and 2 PM. Please let me know if any of these times work for you, or suggest a more convenient time.

                    Thank you for your attention to this matter. I look forward to your response.

                    Best regards,  
                    John Smith  
                    Marketing Manager  
                    XYZ Corporation
                """


    prediction, probabilities = predict(test_text)
    
    spam_prob = probabilities[0][1] * 100  
    ham_prob = probabilities[0][0] * 100  

    print(f"Message: {test_text}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'ham'}")
    print(f"Probabilities - Spam: {spam_prob:.2f}%, ham: {ham_prob:.2f}%")

import pandas as pd
from bs4 import BeautifulSoup
from sklearn import preprocessing
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

import evaluate
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import torch
# nltk.download('punkt')
# nltk.download('stopwords')

data_path = "spam_ham_dataset.csv" 
text_column_name = "text"  
label_column_name = "class"  

df = pd.read_csv(data_path)

df.drop('Unnamed: 0', axis=1, inplace=True)

df.columns = ['label', 'text', 'class']

# Clean the text (removing HTML tags)
class Cleaner():
    def __init__(self):
        pass
    def put_line_breaks(self, text):
        text = str(text).replace('</p>', '</p>\n')
        return text
    def remove_html_tags(self, text):
        cleantext = BeautifulSoup(text, "lxml").text
        return cleantext
    def clean(self, text):
        text = self.put_line_breaks(text)
        text = self.remove_html_tags(text)
        return text

cleaner = Cleaner()
df['text_cleaned'] = df[text_column_name].apply(cleaner.clean)

# Removing stopwords
stop_words = set(stopwords.words('english'))
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if not word in stop_words]))

# Label Encoding
le = preprocessing.LabelEncoder()
le.fit(df[label_column_name].tolist())
df['label'] = le.transform(df[label_column_name].tolist())


X_train, X_test, y_train, y_test = train_test_split(df['text_cleaned'], df['label'], test_size=0.20, random_state=11)


train_df = pd.DataFrame({"text_cleaned": X_train, "label": y_train})
test_df = pd.DataFrame({"text_cleaned": X_test, "label": y_test})

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    return tokenizer(examples["text_cleaned"], truncation=True, padding=True)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)


num_labels = 2  
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model('spam_model')

torch.save(model.state_dict(), 'spam_model.pt')

print("\n--- Evaluation on Training Data ---")
preds_train = trainer.predict(tokenized_train)
preds_train_labels = np.argmax(preds_train.predictions, axis=1)
GT_train = y_train.tolist()
print(classification_report(GT_train, preds_train_labels))

print("\n--- Evaluation on Test Data ---")
preds_test = trainer.predict(tokenized_test)
preds_test_labels = np.argmax(preds_test.predictions, axis=1)
GT_test = y_test.tolist()
print(classification_report(GT_test, preds_test_labels))

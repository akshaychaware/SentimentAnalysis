# from transformers import pipeline

# classifier = pipeline(
#     "sentiment-analysis",
#     model="distilbert-base-uncased-finetuned-sst-2-english"
# )

# text = "I love Python and Django"

# result = classifier(text)

# print(result)

from transformers import pipeline       #vImports the pipeline function from the Hugging Face Transformers library.

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

while True:
    text = input("Enter text: ")

    if text.lower() == "exit":
        break

    result = classifier(text)

    print(result)



    #flow

# Your Text
#    ↓
# Tokenizer
#    ↓
# DistilBERT Model
#    ↓
# Prediction
#    ↓
# Result




#Data Flow Diagram

# User Types:
# "I love cricket"

#        ↓

# input()

#        ↓

# text = "I love cricket"

#        ↓

# classifier(text)

#        ↓

# Tokenizer

#        ↓

# DistilBERT Model

#        ↓

# Prediction

#        ↓

# {'label': 'POSITIVE',
#  'score': 0.9997}

#        ↓

# print()





#Visual Summary

#"I love Python"

#         ↓

# Tokenizer

#         ↓

# [101,1045,2293,18750,102]

#         ↓

# DistilBERT

#         ↓

# Logits

# [-3.45, 5.92]

#         ↓

# Softmax

# [0.00009, 0.99991]

#         ↓

# Prediction

# POSITIVE (99.991%)
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the trained model and tokenizer from the saved directory
model_dir = "model_save/"  
tokenizer = AutoTokenizer.from_pretrained(model_dir)  
model = AutoModelForTokenClassification.from_pretrained(model_dir)  

# Define the mapping 
id2tag = {0: 'O', 1: 'B-MOUNTAIN', 2: 'I-MOUNTAIN'}

def predict(text):
    """
    Makes predictions for tags on the given input text.
    Args:
        text (str): The input text for prediction.
    Returns:
        list: A list of token and label pairs.
    """
    # Tokenize the input text 
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Set the model 
    model.eval()

    # Perform inference without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)  # Forward pass through the model

    # Convert input token IDs back to their corresponding tokens (words/subwords)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    # Get the predicted labels 
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

    # Convert the numeric predictions into their respective label names using the id2tag 
    predicted_labels = [id2tag[pred] for pred in predictions]

    # Remove special tokens like [CLS] and [SEP] from the output
    token_label_pairs = [
        (token, label) for token, label in zip(tokens, predicted_labels)
        if token not in tokenizer.all_special_tokens
    ]

    return token_label_pairs  

# Test the prediction
test_text = "Next on our list is Denali Peak, also known as Mount Everest, in Alaska."
token_label_pairs = predict(test_text)

# Print the prediction results
print(predict(test_text))  

for token, label in token_label_pairs:
    print(f'{token}: {label}')
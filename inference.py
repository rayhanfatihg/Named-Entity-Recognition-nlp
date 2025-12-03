import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_model(model_path):
    # Configuration
    model_name = "indobenchmark/indobert-base-p1"
    
    # Label Mapping from the notebook
    label_to_id = {'O': 0, 'B-Place': 1, 'B-Person': 2, 'I-Place': 3, 'I-Person': 4, 'B-Organisation': 5, 'I-Organisation': 6}
    id_to_label = {i: l for l, i in label_to_id.items()}
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model architecture...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True
    )
    
    # Resize embeddings as done in training
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading weights from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if it's a full model or state_dict
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Debug: Check for classifier weights
            classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
            if not classifier_keys:
                print("WARNING: No classifier weights found in state_dict! Model might be untrained.")
            else:
                print(f"Found classifier weights: {len(classifier_keys)} keys")

            # Handle potential prefix issues
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
                    
            model.load_state_dict(new_state_dict)
            print("Model weights loaded successfully.")
        else:
            print("Loaded object is not a dict, assuming full model...")
            model = state_dict
            print("Full model loaded successfully.")
            
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return None, None, None

    model.eval()
    return model, tokenizer, id_to_label

def predict(model, tokenizer, id_to_label, text):
    print(f"\n--- Debugging Prediction ---")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"Tokens: {tokens}")
    print(f"Input IDs: {inputs['input_ids'][0].tolist()}")

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    
    predicted_ids = predictions[0].tolist()
    print(f"Predicted IDs: {predicted_ids}")
    predicted_labels_raw = [id_to_label[p] for p in predicted_ids]
    print(f"Raw Labels: {predicted_labels_raw}")
    
    # Post-processing to merge subwords and format output
    results = []
    current_word = ""
    current_label = None
    
    print("\n--- Reconstructing Words ---")
    for token, label in zip(tokens, predicted_labels_raw):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        # Handle subwords
        # IndoBERT tokenizer usually uses ## for subwords
        if token.startswith("##"):
            current_word += token[2:]
            # We keep the label of the first piece of the word
        else:
            if current_word:
                results.append((current_word, current_label))
                # print(f"Finished word: {current_word} -> {current_label}")
            current_word = token
            current_label = label
            
    if current_word:
        results.append((current_word, current_label))
        
    return results

if __name__ == "__main__":
    model_path = "IndoBERT.pth"
    text = "Michael jalan ke Bandung untuk makan ."
    
    print(f"Input text: {text}")
    
    model, tokenizer, id_to_label = load_model(model_path)
    
    if model:
        print("\nPrediction Result:")
        entities = predict(model, tokenizer, id_to_label, text)
        print("\nFinal Entities:")
        for word, label in entities:
            print(f"{word}: {label}")

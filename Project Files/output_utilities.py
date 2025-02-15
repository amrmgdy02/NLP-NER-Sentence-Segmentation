
import torch
import os
import pandas
import pipeline_utils as utils
def predict_single_string(text, model, tokenizer, id2tag):

    device = model.device
    
    tokenized = tokenizer(
        text.split(),
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
    inputs = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, axis=2)[0]
       # print(predictions)
    predictions = predictions.cpu()
    
    predicted_tags = []
    word_ids = tokenized.word_ids(0) 
    previous_word_id = None
    
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous_word_id:
            continue
        
        predicted_tags.append(id2tag[predictions[idx].item()])
        previous_word_id = word_id
    
    words = text.split()
    return list(zip(words, predicted_tags))


import pandas as pd
import json
import os

def process_csv_orders(csv_path):
    try:
        # Read CSV with no headers
        df = pd.read_csv(csv_path)
        
        # Get total rows
        total_rows = len(df)
        # Iterate through all rows
        for i in range(total_rows):
            try:
                order_text = df.iloc[i]['order']
                create_json(text=order_text, num=i)
                
            except Exception as e:
                print(f"Error processing row {i}: {str(e)}")
                print(order_text)
                continue
                
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        
        
def create_json(text, num, is_tokenizer, is_model, ner_tokenizer, ner_model):
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"{num}.json")
    
    try:
        # Process the order
        results = process_order_dynamic(
            text,
            is_tokenizer,
            is_model,
            ner_tokenizer,
            ner_model,
            utils.IS_id2tag,
            utils.id2tag
        )
        
        # Create JSON
        order_json = create_order_json(results)
        
    except Exception as e:
        print(f"Error processing order: {num}")
        # Write empty JSON on failure
        order_json = {}
        try:
            with open(output_file, "w") as f:
                json.dump(order_json, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return False
    
    # Save to file (either full or empty JSON)
    try:
        with open(output_file, "w") as f:
            json.dump(order_json, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False
    

def create_order_json(ner_results):
    order_json = {
        "ORDER": {
        }
    }
    
    current_order = None
    order_types = []
    order_labels = []
    order_tokens = []
    for tokens, is_tags, labels in ner_results:
        if is_tags[0] == 'B-PIZZAORDER' or is_tags[0] == 'I-PIZZAORDER':
            current_order = 'PIZZAORDER'
        elif is_tags[0] == 'B-DRINKORDER' or is_tags[0] == 'I-DRINKORDER':
            current_order = 'DRINKORDER'
        order_types.append(current_order)
        order_labels.append(labels)
        order_tokens.append(tokens)
    order_types = list(order_types)
    order_labels = list(order_labels)
    order_tokens = list(order_tokens)
    pizza_orders = []
    drink_orders = []
    # print(order_types)
    # print(order_labels)
    # print(order_tokens)
    for j in range(len(order_types)):
        current_order = None
        if order_types[j] == "PIZZAORDER":
            structured_order = {}
            current_topping = None
            toppings = []
            style = {}
            styles = []
            i = 0
            current_quantity = ""
            while i < len(order_tokens[j]):
                token = order_tokens[j][i]
                tag = order_labels[j][i]
                if token =="dough" and tag == "I-TOPPING":
                    tag = "I-STYLE"
                elif token =="dough" and tag == "I-NOT_TOPPING":
                    tag = "I-NOT_STYLE"
                if tag == "B-NOT_STYLE" and i+1 < len(order_tokens[j]) and order_labels[j][i+1] == "B-NOT_TOPPING":
                    tag = "B-QUANTITY"
                if tag == "B-NUMBER" and not structured_order.get("NUMBER"):
                    structured_order["NUMBER"] = token
                elif tag == "B-NUMBER":
                    if i + 1 < len(order_tokens[j]) and order_labels[j][i + 1] == "I-QUANTITY":
                        current_topping = {
                            "NOT": False,
                            "Quantity": token + " " + order_tokens[j][i + 1],
                            "Topping": None
                        }
                        current_quantity = current_topping["Quantity"]
                        i += 1
                elif tag == "I-NUMBER":
                    if structured_order.get("NUMBER"):
                        structured_order["NUMBER"] += f" {token}"
                    else:
                        structured_order["NUMBER"] = token
                elif tag == "B-SIZE":
                    structured_order["SIZE"] = token
                elif tag == "I-SIZE":
                    if structured_order.get("SIZE"):
                        structured_order["SIZE"] += f" {token}"
                    else:
                        structured_order["SIZE"] = token
                elif tag == "B-STYLE":
                    if style != {}:
                        styles.append(style)
                        style={}
                    style = {"NOT": False, "Style": token}
                elif tag == "I-STYLE":
                    if style == {}:
                        style = {"NOT": False, "Style": token}
                    else:
                        if style.get("Style"): 
                            style["Style"] += f" {token}"
                        else:
                            style = {"NOT": False, "Style": token}
                elif tag == "B-NOT_STYLE":
                    if style != {}:
                        styles.append(style)
                        style={}
                    style = {"NOT": True, "Style": token}
                elif tag == "I-NOT_STYLE":
                    if style == {}:
                        style = {"NOT": True, "Style": token}
                    else:
                        if style.get("Style"):
                            style["Style"] += f" {token}"
                        else:
                            style = {"NOT": True, "Style": token}
                elif tag == "B-TOPPING":
                    # Check if previous token was a quantity
                    current_topping = {
                        "NOT": False,
                        "Quantity": None,
                        "Topping": token
                    }
                    toppings.append(current_topping)
                elif tag == "B-NOT_TOPPING":
                    current_topping = {
                        "NOT": True,
                        "Quantity": None,
                        "Topping": token
                    }
                    toppings.append(current_topping)
                elif  tag == "I-TOPPING" and current_topping:
                    if current_topping == None:
                        current_topping = {
                            "NOT": False,
                            "Quantity": None,
                            "Topping": token
                        }
                        toppings.append(current_topping)
                    elif current_topping.get("Topping"):
                        current_topping["Topping"] += f" {token}"
                    else:
                        current_topping["Topping"] = token
                elif tag == "I-TOPPING" and not current_topping:
                    current_topping = {
                        "NOT": False,
                        "Quantity": None,
                        "Topping": token
                    }
                    toppings.append(current_topping)
                elif tag == "I-NOT_TOPPING" and current_topping:
                    if current_topping == None:
                        current_topping = {
                            "NOT": True,
                            "Quantity": None,
                            "Topping": token
                        }
                        toppings.append(current_topping)
                    elif current_topping.get("Topping"):
                        current_topping["Topping"] += f" {token}"
                    else:
                        current_topping["Topping"] = token
                elif tag == "I-NOT_TOPPING" and not current_topping:
                    current_topping = {
                        "NOT": True,
                        "Quantity": None,
                        "Topping": token
                    }
                    toppings.append(current_topping)
                elif tag == "B-QUANTITY":
                    #look behind to see if previous token was a quantity
                    current_topping = {
                        "NOT": False,
                        "Quantity": token,
                        "Topping": None
                    }
                    current_quantity = current_topping["Quantity"]
                    # Look ahead to see if next token is a topping
                    if i + 1 < len(order_tokens[j]) and order_labels[j][i + 1] == "B-TOPPING" :
                        current_topping["Topping"] = order_tokens[j][i + 1]
                        i += 1  # Skip the next token as we've processed it
                    elif i+1 < len(order_tokens[j]) and order_labels[j][i + 1] == "B-NOT_TOPPING":
                        current_topping["Topping"] = order_tokens[j][i + 1]
                        current_topping["NOT"] = True                      
                        i += 1
                    toppings.append(current_topping)
                elif tag == "I-QUANTITY":
                    if not current_topping:
                        current_topping = {
                            "NOT": False,
                            "Quantity": token,
                            "Topping": None
                        }
                        current_quantity = current_topping["Quantity"]
                    if current_topping.get("Quantity"):
                        current_topping["Quantity"] += f" {token}"
                        current_quantity = current_topping["Quantity"]
                    else:
                        current_topping = {
                            "NOT": False,
                            "Quantity": token,
                            "Topping": None
                        }
                        current_quantity = current_topping["Quantity"]
                    if i + 1 < len(order_tokens[j]) and order_labels[j][i + 1] == "B-QUANTITY":
                        current_topping["Quantity"] += f" {order_tokens[j][i + 1]}"
                        current_quantity = current_topping["Quantity"]
                        i+=1
                        if i + 1 < len(order_tokens[j]) and order_labels[j][i + 1] == "B-TOPPING":
                            current_topping["Topping"] = order_tokens[j][i + 1]
                            toppings.append(current_topping)
                            i += 1
                        elif i+1 < len(order_tokens[j]) and order_labels[j][i + 1] == "B-NOT_TOPPING":
                            current_topping["Topping"] = order_tokens[j][i + 1]
                            current_topping["NOT"] = True
                            toppings.append(current_topping)
                            i += 1
                    elif i + 1 < len(order_tokens[j]) and order_labels[j][i + 1] == "B-TOPPING":
                        current_topping["Topping"] = order_tokens[j][i + 1]
                        toppings.append(current_topping)
                        i += 1
                    elif i+1 < len(order_tokens[j]) and order_labels[j][i + 1] == "B-NOT_TOPPING":
                        current_topping["Topping"] = order_tokens[j][i + 1]
                        current_topping["NOT"] = True
                        toppings.append(current_topping)
                        i += 1
                i += 1
                if toppings != []:
                    structured_order["AllTopping"] = toppings
            if style != {}:
                styles.append(style)
            if styles != []:
                structured_order["STYLE"] = styles
            if not structured_order.get("NUMBER") and structured_order!={}:
                structured_order["NUMBER"] = 1
            pizza_words = {'pizza', 'pizzas', 'pie', 'pies'}
            filtered_tokens = [
                token for token in order_tokens [j]
                if token.lower() in pizza_words
            ]
            if not ((structured_order.get("STYLE") == None and structured_order.get("AllTopping") == None)) or filtered_tokens!=[]:
                if(structured_order!={}):
                    pizza_orders.append(structured_order)
                continue
        if order_types[j] == "DRINKORDER" or order_types[j] == "PIZZAORDER":
            structured_order = {}
            for token, tag in zip(order_tokens[j], order_labels[j]):
                if tag == "B-NUMBER":
                    structured_order["NUMBER"] = token
                elif tag == "I-NUMBER":
                    if structured_order.get("NUMBER"):
                        structured_order["NUMBER"] += f" {token}"
                    else:
                        structured_order["NUMBER"] = token
                elif tag == "B-VOLUME":
                    structured_order["VOLUME"] = token
                elif tag == "I-VOLUME":
                    if structured_order.get("VOLUME"):
                        structured_order["VOLUME"] += f" {token}"
                    else:
                        structured_order["VOLUME"] = token
                elif tag == "B-SIZE":
                    structured_order["SIZE"] = token
                elif tag == "I-SIZE":
                    if structured_order.get("SIZE"):
                        structured_order["SIZE"] += f" {token}"
                    else:
                        structured_order["SIZE"] = token
                elif tag == "B-DRINKTYPE":
                    structured_order["DRINKTYPE"] = token
                elif tag == "I-DRINKTYPE":
                    if structured_order.get("DRINKTYPE"):
                        structured_order["DRINKTYPE"] += f" {token}"
                    else:
                        structured_order["DRINKTYPE"] = token
                elif tag == "B-CONTAINERTYPE":
                    structured_order["CONTAINERTYPE"] = token
                elif tag == "I-CONTAINERTYPE":
                    if structured_order.get("CONTAINERTYPE"):
                        structured_order["CONTAINERTYPE"] += f" {token}"
                    else:
                        structured_order["CONTAINERTYPE"] = token
                
            if not structured_order.get("NUMBER") and structured_order!={}:
                structured_order["NUMBER"] = 1 
            if(structured_order!={}):
                drink_orders.append(structured_order)
    if pizza_orders != []:
        order_json["ORDER"]["PIZZAORDER"] = pizza_orders
    if drink_orders != []:   
        order_json["ORDER"]["DRINKORDER"] = drink_orders
    return order_json

def process_order_dynamic(text, IS_tokenizer, IS_model, NER_tokenizer, NER_model, IS_id2tag, NER_id2tag):
    tokens = text.split()
    device = IS_model.device
    
    IS_inputs = IS_tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    word_ids = IS_inputs.word_ids(batch_index=0)
    inputs = {k: v.to(device) for k, v in IS_inputs.items()}
    
    # Get IS predictions
    with torch.no_grad():
        IS_outputs = IS_model(**inputs)
        IS_predictions = torch.argmax(IS_outputs.logits, dim=-1)
    
    # Group tokens by continuous non-zero IS predictions
    sequences = []
    current_sequence = []
    current_tokens = []
    current_is_tags = []
    previous_word_id = None
    
    for word_id, pred in zip(word_ids, IS_predictions[0]):
        if word_id is None:
            if current_sequence:
                sequences.append((current_tokens.copy(), current_sequence.copy(), current_is_tags.copy()))
                current_sequence = []
                current_tokens = []
                current_is_tags = []
            continue
            
        if word_id != previous_word_id:  # Only process new words
            pred_val = pred.item()
            if pred_val != 0:  # If not 'O' tag
                current_sequence.append(pred_val)
                current_tokens.append(tokens[word_id])
                current_is_tags.append(IS_id2tag[pred_val])
            else:
                if current_sequence:
                    sequences.append((current_tokens.copy(), current_sequence.copy(), current_is_tags.copy()))
                    current_sequence = []
                    current_tokens = []
                    current_is_tags = []
        previous_word_id = word_id
    
    # Add last sequence if exists
    if current_sequence:
        sequences.append((current_tokens.copy(), current_sequence.copy(), current_is_tags.copy()))
    
    # Process each sequence through NER
    results = []
    for seq_tokens, _, is_tags in sequences:
        if seq_tokens:
            NER_inputs = NER_tokenizer(
                seq_tokens,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            ner_inputs = {k: v.to(device) for k, v in NER_inputs.items()}
            
            with torch.no_grad():
                NER_outputs = NER_model(**ner_inputs)
                NER_predictions = torch.argmax(NER_outputs.logits, dim=-1)
            
            previous_word_id = None
            ner_labels = []
            
            for word_id, pred in zip(NER_inputs.word_ids(batch_index=0), NER_predictions[0].cpu().numpy()):
                if word_id is None:
                    continue
                if word_id != previous_word_id:  # Only process new words
                    ner_labels.append(NER_id2tag[pred])
                previous_word_id = word_id
            
            results.append((seq_tokens, is_tags, ner_labels ))
    return results
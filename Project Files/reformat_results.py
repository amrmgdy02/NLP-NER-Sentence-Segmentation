import json
import os
import pandas as pd
import argparse 

def keys_to_upper(sub_tree: dict):
    sub_tree = {key.upper(): value for key, value in sub_tree.items()}
    return sub_tree


def process_pizzas(pizza_list):
    output_str = ""
    for pizza in pizza_list:
        output_str += "(PIZZAORDER "
        pizza = keys_to_upper(pizza)

        if "NUMBER" in pizza.keys() and pizza["NUMBER"]:
            output_str += f"(NUMBER {pizza["NUMBER"]} ) "

        if "SIZE" in pizza.keys() and pizza['SIZE']:
            output_str += f"(SIZE {pizza["SIZE"]} ) "

        if "STYLE" in pizza.keys() and pizza['STYLE']:
            for style in pizza['STYLE']:
                style = keys_to_upper(style)
                negated_style = False
                if "NOT" in style.keys() and style['NOT']:
                    output_str += "(NOT "
                    negated_style = True
                if "TYPE" in style.keys() and style['TYPE']:
                    output_str += f"(STYLE {style['TYPE']} ) "
                if negated_style:
                    output_str += ") "

        if "ALLTOPPING" in pizza.keys() and pizza['ALLTOPPING']:
            for topping in pizza['ALLTOPPING']:
                
                topping = keys_to_upper(topping)

                negated_topping = False
                if "NOT" in topping.keys() and topping['NOT']:
                    output_str += "(NOT "
                    negated_topping = True

                if "QUANTITY" in topping.keys() and topping['QUANTITY']:
                    output_str += f"(COMPLEX_TOPPING (QUANTITY {topping['QUANTITY']} ) "
                    if "TOPPING" in topping.keys() and topping['TOPPING']:
                        output_str += f"(TOPPING {topping['TOPPING']} ) "
                    output_str += ") "
                    # continue
                
                elif "TOPPING" in topping.keys() and topping['TOPPING']:
                    output_str += f"(TOPPING {topping['TOPPING']} ) "
                    
                if negated_topping:
                    output_str += ") "
        output_str += ") "
    return output_str

def process_drinks(drinks_list):
    output_str = ""
    for drink in drinks_list:
        output_str += "(DRINKORDER "
        drink = keys_to_upper(drink)

        if "NUMBER" in drink.keys() and drink["NUMBER"]:
            output_str += f"(NUMBER {drink["NUMBER"]} ) "

        if "SIZE" in drink.keys() and drink['SIZE']:
            output_str += f"(SIZE {drink["SIZE"]} ) "

        if "DRINKTYPE" in drink.keys() and drink['DRINKTYPE']:
            output_str += f"(DRINKTYPE {drink['DRINKTYPE']} ) "

        if "VOLUME" in drink.keys() and drink['VOLUME']:
            output_str += f"(VOLUME {drink['VOLUME']} ) "

        if "CONTAINERTYPE" in drink.keys() and drink['CONTAINERTYPE']:
            output_str += f"(CONTAINERTYPE {drink['CONTAINERTYPE']} ) "

        output_str += ") "
    return output_str

def parse_tree(tree: dict):
    tree = keys_to_upper(tree)
    output_str = ""
    if "ORDER" in tree.keys():
        output_str += "(ORDER "
        tree = tree['ORDER']

        tree = keys_to_upper(tree)

        pizzas = []
        if 'PIZZAORDER' in tree.keys():
            pizzas = tree['PIZZAORDER']

        drinks = []
        if 'DRINKORDER' in tree.keys():
            drinks = tree['DRINKORDER']

        output_str += process_pizzas(pizzas)

        output_str += process_drinks(drinks)

        output_str += ")"
    return output_str


# Driver code:
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_folder', type=str, help='json folder path')
    args = parser.parse_args()

    folder_path = args.json_folder
    outs = []
    for j in os.listdir(folder_path):
        with open(os.path.join(folder_path, j), 'r') as j_file:
            test_tree = json.load(j_file)
        out = parse_tree(test_tree)
        id = j.split('.')[0]
        outs.append({'id': id, 'output': out})

    pd.DataFrame(outs).to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()

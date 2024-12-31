import requests
import json
from kiwipiepy import Kiwi

class KiwiTokenizer():
    def __init__(
            self, 
            kiwi_load_default_dict:bool=True,
            kiwi_model_type:str="sbg",
        ):
        self.kiwi = Kiwi(num_workers=0, load_default_dict=kiwi_load_default_dict, model_type=kiwi_model_type)
        self.kiwi.tokenize("")

    def run(self, text:str, saisiot:bool=True) -> list[str]:
        result = self.kiwi_tokenizer.tokenize(text, saisiot=saisiot, split_complex=True)
        token_forms = [token.form for token in result]
        return token_forms

    def run_for_bbpe(self, text:str, saisiot:bool=True) -> str:
        token_forms = self.run(text, saisiot=saisiot)
        tokenized_item = " ".join(token_forms)
        return tokenized_item

def nori_NNP(text:str, is_print:bool=False):
    endpoint = "http://15.164.151.11:9200"
    index = "goods-ko"
    url = f"{endpoint}/{index}/_analyze"
    auth = ("admin", "X2commerce!1")

    payload = {
        "explain": True,
        "tokenizer": "nori_tokenizer",
        "text": [text]
    }

    try:
        response = requests.post(url, json=payload, auth=auth)
        response.raise_for_status()
        response_data = response.json()
        tokens = response_data.get('detail', {}).get('tokenizer', {}).get('tokens', [])
        filtered_tokens = [
            token for token in tokens
            if token.get('leftPOS') in ["NNG(General Noun)", "NNP(Proper Noun)"]
        ]

        token_list = [token['token'] for token in filtered_tokens]
        if is_print:
            print(token_list)
        return token_list
    except requests.exceptions.RequestException as e:
        print(f"Error Occur: {e}")


# if __name__ == "__main__":
#     while True:
#         try:
#             user_input = input("User: ")
#             if user_input.lower() in ["quit", "exit", "q"]:
#                 print("Goodbye!")
#                 break

#             nori_NNP(user_input, is_print=True)
#         except:
#             print("Error Occur")
#             break
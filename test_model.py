import os
os.chdir("/workspace")
from models.bert_cls import plateer_classifier;

if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            result = plateer_classifier(user_input)[0]
            print("-----------Category-----------")
            for item in result:
                print(item)
            print("")
        except:
            print("Error Occur")
            break

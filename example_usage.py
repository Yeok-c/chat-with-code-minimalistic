from repo_chat import RepoChat
import os

if __name__ == "__main__":
    repollm = RepoChat(
        codebase_path="https://github.com/Yeok-c/Stewart_Py",
        # model_name="gpt-3.5-turbo-16k", #default, can be changed
        # suffixes=[".py", ".ipynb"]  # default, can be changed
        )
    
    try:
        while True:
            question = input("Input: ")
            answer = repollm.chat(question)
            print("Output: " + answer)
            print("\n")

    
    except KeyboardInterrupt:
        print(' \nExiting program')
        repollm.print_usage()
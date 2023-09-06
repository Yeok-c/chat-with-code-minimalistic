from repo_chat import RepoChat
import os

if __name__ == "__main__":
    repollm = RepoChat(
        "https://github.com/Yeok-c/Stewart_Py",
        db_creation_mode="github",
        # model_name="gpt-3.5-turbo", #default, can be changed
        # suffixes=[".py", ".ipynb", ".md"]  # default, can be changed
        )
    
    try:
        while True:
            question = input("Input: ")
            answer = repollm.chat(question)
            print("\nOutput: " + answer + "\n\n")
    
    except KeyboardInterrupt:
        print(' \nExiting program')
        repollm.print_usage()
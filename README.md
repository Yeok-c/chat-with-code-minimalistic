# RepoChat - Chat with you github repository or offline codebase
 Minimal langchain/python code to query code or semi-automatically write code
    - Load codebase into Langchain using Chroma's vectorstore
    - Search through vectorstore using ConversationalRetrievalChain
    - Retain conversational memory during chat with ConversationSummaryMemory

# Installation
```git clone https://github.com/Yeok-c/repo-chat/```
```cd repo-chat```
```pip install -r requirements.txt```

Ready to run
```python example.py```

Example output:



# If you have not set up OpenAI API keys in environment variables
In your terminal
```nano ~/.bashrc```

Navigate to the bottom and add this line
```export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx # your openai-api key```

Exit nano and
``` source ~/.bashrc```

You may need to reactivate your conda environment after that

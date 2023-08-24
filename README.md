# RepoChat - Chat with you github repository or offline codebase
Minimal langchain/python code to query code or semi-automatically write code
   - Load codebase into Langchain using Chroma's vectorstore
   - Search through vectorstore using ConversationalRetrievalChain
   - Retain conversational memory during chat with ConversationSummaryMemory

# Installation
```git clone https://github.com/Yeok-c/repo-chat/```
```cd repo-chat```
```pip install -r requirements.txt```

## Usage
```python example.py```

# Example output:
````
Cloning from https://github.com/Yeok-c/Stewart_Py to ./temp_repo/
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 1430.20it/s]
Loaded 6 documents, split into 24 texts. 
Loading into chroma. This may take a while.
Repo loaded into langchain.

Based on the information provided in the notebook, the repo appears to be a Python implementation and step-by-step explanation of the inverse kinematics of a Stewart Platform. The main program is called "Stewart_Platform.py" and it contains the code for solving the inverse kinematics problem for a Stewart Platform. The code includes a class called "Stewart_Platform" that takes in parameters for the Stewart Platform configuration (such as the radius of the base and platform, servo horn length, rod length, etc.) and provides a method called "calculate" to solve for the rotation angles of the servos given a translation vector and a rotation vector.

The repo also includes a plotting function called "plot_platform" that visualizes the Stewart Platform in a 3D plot. It shows the base, the rods, and the platform, and allows you to see how the platform moves based on the servo angles.

The readme file provides additional information about the project and its usage. It explains how to import the package, create an object of the Stewart_Platform class, and use the "calculate" method to solve for the servo angles. It also includes links to additional resources and implementations of Stewart Platforms in other programming languages.

Overall, the repo seems to be a comprehensive tutorial and implementation of the inverse kinematics of a Stewart Platform in Python.


Input: Based on01_Stewart_Py_Inverse_Kinematics.ipynb how do I create a stewart platform python object.        
Output: The contents of the `01_Stewart_Py_Inverse_Kinematics.ipynb` notebook include a tutorial on solving the inverse kinematics of a Stewart platform using Python. The notebook provides step-by-step explanations and code examples for calculating the servo angles of a Stewart platform.

To create a Stewart Platform Python object, you need to follow these steps:

1. Import the `Stewart_Platform` class from the `stewart_controller` module.

```python
from stewart_controller import Stewart_Platform
```

2. Instantiate the `Stewart_Platform` object by providing the necessary parameters for the Stewart platform. The parameters include the radius of the base (`r_B`), radius of the platform (`r_P`), servo horn length (`lhl`), rod length (`ldl`), and the half-angle between two anchors on the base (`Psi_B`) and platform (`Psi_P`).

```python
platform = Stewart_Platform(r_B, r_P, lhl, ldl, Psi_B, Psi_P)
```

3. Use the `calculate` method of the `platform` object to solve for the inverse kinematics and obtain the servo angles. The `calculate` method takes two arguments: the translation vector (`[tx, ty, tz]`) and the rotation vector (`[rx, ry, rz]`).

```python
servo_angles = platform.calculate(np.array([tx, ty, tz]), np.array([rx, ry, rz]))
```

4. Finally, you can plot the Stewart platform using the `plot_platform` method of the `platform` object.

```python
ax = platform.plot_platform()
plt.show()
```

Note that the code provided is a condensed version of the tutorial, and it assumes that you have already defined the necessary variables and imported the required libraries.
````


# If you have not set up OpenAI API keys in environment variables
In your terminal
```nano ~/.bashrc```

Navigate to the bottom and add this line
```export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx # your openai-api key```

Exit nano and
``` source ~/.bashrc```

You may need to reactivate your conda environment after that

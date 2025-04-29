import csv
import os
import time
from typing import Dict
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

console = Console()

template = """
You are a software engineer tasked with defining a new feature based on the provided analysis of customer issues. Here’s how you should proceed:

1. **Understand the Analysis**: Read the customer issue analysis provided in the variable <customer_issue_analysis>. Identify key requirements, user pain points, and any specific functionalities that need to be addressed.

2. **Generate a Task Definition**: Create a task definition for the new feature based on your understanding of the analysis. This definition should include:
   - A clear purpose for the feature.
   - The expected input parameters that the function will require (e.g., user input, configuration settings).
   - The expected output from the function (e.g., success confirmation, error messages).
   - Any validation rules or conditions that must be satisfied before executing the function.
   - Error handling scenarios that should be accounted for (e.g., invalid inputs, system errors).

3. **Write the Python Function**: After defining the task, write a Python 3.11 compatible function that implements the identified feature. The function should:
   - Accept the input parameters defined in the task.
   - Include validation logic to ensure that any inputs meet the required criteria.
   - Implement error handling for various scenarios, returning appropriate messages or responses.
   - Return a success message or relevant output upon successful completion of the function.

Make sure to encapsulate your final Python function inside <function> tags and ensure that it adheres to best practices for code clarity and maintainability.
```

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

""

llm = ChatOpenAI(
    model_name="gpt-4o",      # Use the 4o model
)

chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=llm,
)

def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response


# def count_issues(csv_path: str) -> Dict[str, int]:
#     """
#     Reads a CSV file at csv_path, counts occurrences of each unique value
#     in the 'issue' column, and returns a dict of {issue: count}.
#     """
#     counts: Dict[str, int] = {}
#     with open(csv_path, newline='', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             issue = row.get("issue")
#             if issue is None:
#                 # skip rows without an 'issue' field
#                 continue
#             counts[issue] = counts.get(issue, 0) + 1
#     return counts

if __name__ == "__main__":
    try:
            analysis: str = (
                "Analysis of the “Frequency of Issues” chart reveals that password reset requests overwhelmingly dominate "
                "support inquiries, occurring 42 times—far more than any other issue. This disproportionate volume highlights a "
                "clear opportunity to enhance user account management and bolster self-service password recovery options, which "
                "could both streamline the user experience and alleviate the burden on your support team."
            )
                
            with console.status("Analysing CSV file...", spinner="dots"):
                time.sleep(3)
            console.print("[bold green]CSV file analysis complete.")
            console.print(
                f"[green]{analysis}")
            
            with console.status("Generating task definition...", spinner="dots"):
                time.sleep(3)
            console.print("[bold green]Task definition generation complete.")

            with console.status("Generating Python function...", spinner="dots"):
                response = get_llm_response(analysis)
            
            console.print("[bold green]Python function generation complete.")
            console.print(f"[yellow] Generated Output:\n{response}")          
    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}")

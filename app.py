import os
import time
import streamlit as st
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

console = Console()

template = """
You are a software engineer tasked with defining a new feature based on the provided analysis of customer issues. Here’s how you should proceed:

1. **Understand the Analysis**: Read the customer issue analysis provided. Identify key requirements, user pain points, and any specific functionalities that need to be addressed.

2. **Generate a Task Definition**: Create a structured task definition in markdown for the new feature, including the following elements, though all are not mandatory:

   - **Issue Metadata**:
     - **Issue Type**: Determine the appropriate type (e.g., Task, Story, Bug, Spike).
     - **Project & Component**: Identify the project and any relevant components or modules. For now assume that the project is "User Account Management".
     - **Epic Link**: If applicable, link it to a parent Epic.
     - **Labels**: Add any relevant tags for filtering (e.g., ui, backend).

   - **Summary**: Write a concise, imperative title for the feature, such as “Implement OAuth2 login endpoint for mobile clients”.

   - **Description**: Provide a clear narrative:
     - **Context / Background**: Explain why this work matters (e.g., security requirement, new product need).
     - **Objective**: State what you are trying to achieve.
     - **Approach / Details**: Outline the intended technical approach, including references to documentation, wireframes, or API specs.
     - **User-Story Format** (optional): If appropriate, frame it as “As a [role], I want [capability] so that [benefit].”

   - **Acceptance Criteria**: Create a checklist of testable conditions using “Given/When/Then” style:
     - Example: Given a valid user token, when they POST to /api/login, then they get a 200 response with a JWT.

   - **Estimates & Planning**:
     - **Story Points (or time estimate)**: Provide a relative sizing (e.g., 3 pts) or time estimate (e.g., 2 days).
     - **Sprint / Fix Version**: Indicate which sprint or release this should go into.
     - **Priority**: Assign a priority level (Highest, High, Medium, Low).

   - **Assignment & Workflow**:
     - **Assignee**: Specify the person responsible for implementation.
     - **Reporter**: Identify who created the ticket.
     - **Status / Transitions**: Define the workflow (e.g., To Do → In Progress → In Review → Done).

   - **Technical Notes & Attachments**:
     - **Implementation Details**: Include notes on library choices, endpoint signatures, or data-model diagrams.
     - **Dependencies**: List any dependencies (e.g., “Blocks” / “Is blocked by” links to other issues).
     - **Attachments**: Include any relevant mockups, API docs, logs, or specs.

   - **Definition of Done**: Create a quick checklist ensuring quality and completeness (e.g., code written, peer-reviewed, merged, unit/integration tests added, documentation updated, deployed to staging and smoke-tested).

3. **Write the Python Function**: After defining the task, write a Python 3.11 compatible function that implements the identified feature. The function should:
   - Accept the input parameters defined in the task.
   - Include validation logic to ensure that any inputs meet the required criteria.
   - Implement error handling for various scenarios, returning appropriate messages or responses.
   - Return a success message or relevant output upon successful completion of the function.

Make sure to encapsulate your final Python function inside <function> tags, is in markdown format and ensures that it adheres to best practices for code clarity and maintainability.
```

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:

"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

task_defninition_input: str = (
        "The following below is the output of Step 1.\n"
        "Analysis of the “Frequency of Issues” chart reveals that password reset requests overwhelmingly dominate "
        "support inquiries, occurring 42 times—far more than any other issue. This disproportionate volume highlights a "
        "clear opportunity to enhance user account management and bolster self-service password recovery options, which "
        "could both streamline the user experience and alleviate the burden on your support team.\n"
        "proceed to Step 2. Do not proceed to Step 3.\n"
    )

code_generation_input: str = (
    "The following below is the output of Step 2.\n"
    "{task_definition}"
    "proceed to Step 3.\n"
)


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
    Generates a response to the given text using the large language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response


# 1) Function to simulate “user story task definition”
def generate_user_story_task(task_definition_str: str = task_defninition_input):
    return get_llm_response(task_definition_str)


# 2) Function to simulate “code generation”
def generate_code_snippet(task_definition: str, code_generation_input: str = code_generation_input):
    # Simulate a code generation process
    code_generation_input = code_generation_input.format(task_definition=task_definition)
    return get_llm_response(code_generation_input)


def main(): 
    st.title("PDLC - Product Development Life Cycle Accelerator POC Demo")
    st.subheader("Analyze User Interactions Log")
    
        # Initialize session-state flags
    if "user_story_done" not in st.session_state:
        st.session_state.user_story_done = False
    if "code_gen_done" not in st.session_state:
        st.session_state.code_gen_done = False
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    # —————————————————————————————————————
    # 1. Show a progress bar from 0 → 100 over 5 seconds
    # —————————————————————————————————————
    if not st.session_state.analysis_done:
        with st.spinner("Analyzing User Interactions Log…"):
            time.sleep(1) # Simulate a delay
        st.success("✅ **Analysis Complete!**")
        st.session_state.analysis_done = True
    
    # —————————————————————————————————————
    # 2. Once done, show a paragraph of text
    # —————————————————————————————————————

    if st.session_state.analysis_done:
        st.write(
            "Analysis of the “Frequency of Issues” chart reveals that password reset requests overwhelmingly dominate "
            "support inquiries, occurring 42 times—far more than any other issue. This disproportionate volume highlights a "
            "clear opportunity to enhance user account management and bolster self-service password recovery options, which "
            "could both streamline the user experience and alleviate the burden on your support team."
        )
        
    st.subheader("Define User Story Task")
    # —————————————————————————————————————
    # 3. Button → call define_user_story_task()
    # —————————————————————————————————————
    if not st.session_state.user_story_done:
        if st.button("Proceed to User Story Task Definition"):
            with st.spinner("Generating User Story Task Definition..."):
                time.sleep(1) # Simulate a delay
                st.session_state.user_story_result = generate_user_story_task()
                #console.print(st.session_state.user_story_result)
                st.session_state.user_story_done = True
            st.success("✅ **User Story Generated!**")

    # 4. Print the return of define_user_story_task() as Markdown
    if st.session_state.user_story_done:
        st.markdown(st.session_state.user_story_result)

        st.subheader("Generate Python Function")
        # —————————————————————————————————————
        # 5. Show next button → call generate_code_snippet()
        # —————————————————————————————————————
        if not st.session_state.code_gen_done:
            if st.button("Proceed to Code Generation"):
                with st.spinner("Generating Python Function..."):
                    time.sleep(1) # Simulate a delay
                    st.session_state.code_gen_result = generate_code_snippet(st.session_state.user_story_result)
                    #console.print(st.session_state.code_gen_result)
                    st.session_state.code_gen_done = True
                st.success("✅ **Python Function Generated!**")

    # 6. Print the return of generate_code_snippet() as Markdown
    if st.session_state.code_gen_done:
        st.markdown(st.session_state.code_gen_result)


if __name__ == "__main__":
    # Uncomment the following line to run the Streamlit app
    main()
# Math Tutor Project

Develop a Math Tutor using Retrieval-Augmented Generation (RAG) or agent-based logic to generate tasks, quizzes, and corresponding answers. The system will produce tasks based on a given user input (e.g. topic name) and generate tasks with solutions it deems correct. A small evaluation dataset should be created to assess the quality and correctness of generated content. The dataset will contain fields: input (e.g., math topic), output (the generated task and answer), type (task/quiz), and expected answer. All generated text should be in Ukrainian.

## Examples

1. Use some Math books as context to better generate some tasks about different topics.
2. Use Wolfram Alpha as a tool for Agent.

## Evaluation

Evaluate the system by comparing the generated answers with expected solutions. Focus on both the quality of questions and the accuracy of answers. Iterate on the model to improve performance, ensuring higher accuracy and diversity in question generation.

## Explanation and Iterations

Document the reasoning behind each experiment, including why specific models or logic were chosen, and the improvements made after each iteration. Explain why the solution is effective and suggest further enhancements.

**Note:** A simple agent call or a basic/vanilla RAG setup can contribute **no more than 50% of the total score**, even if all other requirements are fully met.

## Bonus Task - Multi-Agent System

Design a multi-agent system where different agents handle different subtasks. Explain the architecture, why it's structured this way, and how it compares to the previous approach. Analyze the results, highlighting any improvements in performance or user experience.

## Outcome

Jupyter notebook with experiments; JSONL with dataset; all other files which are required to reproduce your code.

## Additional Homework Outcome

If all will be good, we will try to aggregate all generated questions to create a new Ukrainian Math Question Generation dataset.
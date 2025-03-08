

prompt1 = """
Task: Generate similar math question-and-answer pairs based on the provided reference sample. The generated Q&A should meet the following criteria:

1. **Wide Range of Fields:** The questions should cover diverse mathematical fields, such as algebra, geometry, probability, statistics, calculus, and number theory.
2. **Detailed and Rigorous Reasoning:** For each question, provide a clear, step-by-step solving process. Ensure that each step is mathematically rigorous and logically consistent.
3. **Final Answer Format:** The final answer must be presented exactly in the format `###{{<result>}}`, where `<result>` is the correct numerical answer.
4. **Strict JSON Format:** Output the results in JSON format, with each entry containing two keys: `"question"` and `"answer"`. The `"answer"` field should include the entire reasoning process, concluding with the final answer in the specified format.

**Reference Sample:**

{data}
Your task: Please generate at least 3 distinct Q&A pairs following the guidelines above. Each generated Q&A pair should:

Be diverse in terms of mathematical topics.
Provide a complete, step-by-step explanation for the solution.
End the reasoning with the final answer strictly formatted as ###{{<result>}}.
Generate the output strictly in valid JSON format like reference sample.

"""


prompt2 = """ 
Task: Generate Similar Math Questions and Answers

You are tasked with generating **1 math question** and its corresponding **correct answer** in the required format. Ensure that:
1. The generated question and answer simulate the style and structure of the given reference data.
2. The answer is logically and mathematically correct.
3. The final answer in the `answer` field is formatted as `###{{89}}`.
4. The output strictly follows the specified JSON format.

---

### Reference Data:
{data}

---

### Required Output Format:
```json
{{
    "question": "Tom is planning a party and buys 3 large pizzas and 1 small pizza. A large pizza has 16 slices and a small pizza has 8 slices. If his guests eat all the pizza, how many pieces do they eat in total?",
    "answer": "To find the total number of slices, we need to add the number of slices from the large pizzas and the small pizza. 3 large pizzas have 3 x 16 = 48 slices. 1 small pizza has 1 x 8 = 8 slices. So, the total number of slices is 48 + 8 = 56. Answer is ###{{56}}."
}}
```
"""

prompt3 = """
Task: Based on the following example, generate a new math question and its answer. The answer should include a step-by-step explanation and end with the final answer in the format ###{{result}}.

Example:
{data}
Now, please generate one similar Q&A pair in valid JSON format with keys \"question\" and \"answer\".
"""
def get_prompt(name):
    prompt = {}
    prompt["prompt1"] = prompt1
    prompt["prompt2"] = prompt2
    prompt["prompt3"] = prompt3
    return prompt[name]
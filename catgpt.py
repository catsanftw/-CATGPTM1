import os
import tkinter as tk
from openai import OpenAI
from crewai import Agent, Task, Crew

# Define the OpenAI agent
class LMStudioAgent(Agent):
    def __init__(self):
        super().__init__(
            role="AI Autonomous Chatbot",  # Updated role name
            goal="Respond in rhymes to the given prompt.",
            backstory="An AI that always responds with rhyming answers."
        )
    
    def perform_task(self, task: Task):
        # Initialize OpenAI client with LM Studio server setup
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        # Create the system message to ensure the AI understands to rhyme
        system_message = {
            "role": "system",
            "content": "Respond to the following input by creating a response that rhymes with it."
        }

        # Create the user message with the input prompt
        user_message = {
            "role": "user",
            "content": task.description
        }

        # Call the LM Studio API to get a rhyming response
        completion = client.chat.completions.create(
            model="model-identifier",
            messages=[system_message, user_message],
            temperature=0.7,
        )

        # Return the rhyming result of the task
        return completion.choices[0].message['content']

# Define the task
class RhymeTask(Task):
    def __init__(self, prompt):
        super().__init__(
            description=prompt,
            expected_output="A response that rhymes with the input."
        )

# Create the Tkinter GUI
class CatWebGPTApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.title("CatWebGPT 1.0")
        self.geometry("600x400")

        # Create GUI elements
        self.prompt_label = tk.Label(self, text="Enter your prompt:")
        self.prompt_label.pack(pady=10)

        self.prompt_entry = tk.Entry(self, width=50)
        self.prompt_entry.pack(pady=10)

        self.submit_button = tk.Button(self, text="Submit", command=self.submit_prompt)
        self.submit_button.pack(pady=10)

        self.result_text = tk.Text(self, wrap="word", height=10, width=70)
        self.result_text.pack(pady=10)

    def submit_prompt(self):
        # Get the prompt from the entry box
        prompt = self.prompt_entry.get()

        # Create the agent
        agent = LMStudioAgent()

        # Create the task with the provided prompt
        task = RhymeTask(prompt=prompt)

        # Assemble the crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )

        # Execute the task
        result = crew.kickoff()

        # Display the result in the text box
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

# Run the Tkinter app
if __name__ == "__main__":
    app = CatWebGPTApp()
    app.mainloop()

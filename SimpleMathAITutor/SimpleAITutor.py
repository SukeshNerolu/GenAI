import openai

# Set your OpenAI API key
filepath_key = "Open_API_Key.txt"

with open(filepath_key, "r") as f:
    text = f.read()


openai.api_key = text

filepath_prompt = "Prompt.txt"


with open(filepath_prompt, "r") as f:
    system_message = ' '.join(f.readlines())
print(system_message)
# AI Tutor
# Enter exit to terminate the program
max_conversations = 20 
conversation_length = 0
message_history = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Help me solve the equaltion 3x - 9 =21."},
    {"role": "assistant", "content": "Sure! Try moving the 9 to the right hand side of the equation. What do you get?"},
    {"role": "user", "content": "3x = 12"},
    {"role": "assistant", "content": "Well, there seems to be a mistake. When you move 9 to the right hand side, you need to change its sign. Can you try again?"},
    {"role": "user", "content": "3x = 30"},
    {"role": "assistant", "content": "That looks good. great job! Now, try to divide both sides by 3. What do you get?"},
    {"role": "user", "content": "x = 10"},
    {"role": "assistant", "content": "Great Job!"}]
while conversation_length < max_conversations:
    print("I am here!")
    user_input = input()
    # exit if user enters exit
    if "exit" in user_input.lower():
        print("AI Tutor: Exiting the program!")
        break
    message_history.append({"role": "user", "content": user_input})
    # print(message_history)
    chat_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = message_history)
    converse = chat_response.choices[0].message.content
    print("\n", "AI Tutor:")
    # print(chat_response)
    print(converse)
    print("\n")
    message_history.append({"role": "assistant", "content": converse})
    conversation_length = conversation_length + 1
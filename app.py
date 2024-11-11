import openai
import openai
import json
import pandas as pd
from tenacity import wait_random_exponential, retry, stop_after_attempt

# Set your OpenAI API key
filepath_key = "Open_API_Key.txt"

with open(filepath_key, "r") as f:
    text = f.read()

openai.api_key = text

def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    If products are provided, it will include a brief summary of each product in the response.
    '''
    delimiter = "####"

    base_system_message = f"""
    {delimiter}
    You are an intelligent laptop gadget expert and your goal is to find the best laptop for a user.
    You need to ask relevant questions and understand the user profile by analyzing the user's responses.
    Use function calling when you get all the information about the GPU intensity, Display quality, Portability, Multitasking, Processing speed, and Budget.
    The values for all keys, except 'Budget', should be 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by the user.
    {delimiter}

    {delimiter}
    Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalized:
    - The values for all keys, except 'Budget', should strictly be either 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by the user.
    - The value for 'Budget' should be a numerical value extracted from the user's response.
    - 'Budget' value needs to be greater than or equal to 25000 INR. If the user says less than that, please mention that there are no laptops in that range.
    - Do not randomly assign values to any of the keys.
    - The values need to be inferred from the user's response.
    {delimiter}

    {delimiter}
    Thought 1: Ask a question to understand the user's profile and requirements. \n
    If their primary use for the laptop is unclear, ask follow-up questions to understand their needs.
    You are trying to fill the values of all the keys {{'GPU intensity', 'Display quality', 'Portability', 'Multitasking', 'Processing speed', 'Budget'}} in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    If the necessary information has been extracted, only then proceed to the next step. \n
    Otherwise, rephrase the question to capture their profile clearly. \n
    {delimiter}

    {delimiter}
    Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys.
    Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    If yes, move to the next Thought. If no, ask questions on the keys whose values you are unsure of. \n
    It is a good practice to ask questions with sound logic as opposed to directly citing the key you want to understand the value for.
    {delimiter}

    {delimiter}
    Thought 3: Phrase questions logically based on the user's profile to gather details about their preferences for GPU, display, portability, processing, and multitasking without directly asking if they want high, medium, or low levels for these features.
    {delimiter}

    {delimiter}
    Thought 4: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions.
    {delimiter}

    {delimiter}
    Do not ask questions about GPU, multitasking, processing, and display all at once. Keep it short and avoid overwhelming the user.
    Ask a simple question to gather information about one key, then move on to the next.
    Remember not to ask too many questions at once. You will be heavily penalized otherwise.
    {delimiter}

    {delimiter}
    Once all the necessary information has been gathered and you are confident about the values for the different keys, call the function `laptop_spec` with the collected responses to populate the python dictionary.
    {delimiter}

    {delimiter}
    Here is a sample conversation between the user and assistant:
    User: "Hi, I am a graphic designer."
    Assistant: "Welcome! As a graphic designer, you’ll need a laptop that can handle creative software smoothly and deliver vibrant visuals. This means a high-resolution display and strong multitasking capability are essential. Could you tell me more about your main work tools? Do you primarily use software like Photoshop, Illustrator, or any 3D design applications? Understanding the type of design work you do will help me recommend the right specifications. Let me know if I’m on the right track!"

    User: "I mainly work with Photoshop and Illustrator."
    Assistant: "Thanks for that info! Since you work with Photoshop and Illustrator, a high-resolution display will be essential for accurate color representation. You’ll also need strong multitasking capability to work with large files and multiple layers. Do you frequently work with large, high-resolution files or complex designs? This will help me gauge the level of processing power and storage you’ll need."

    User: "Yes, I work with high-res images often."
    Assistant: "Got it. High-res images demand strong processing power and plenty of storage. For clarity, I have one more question: Do you need a laptop that's easy to carry around, or do you mostly work from one location?"

    User: "Mostly from one location."
    Assistant: "Understood! Just one last question to finalize your preferences: Could you share your budget range for this laptop? This will help me find options that fit your needs within your price range."

    User: "My budget is up to 1.5 lakh INR."
    Assistant: [Function Call: laptop_spec with the collected responses]
    {delimiter}

    {delimiter}
    After gathering the user's information, the best recommended laptops will be fed to you in the user message.
    Keep the user's profile in mind and address any queries using the catalog in the user message.
    {delimiter}

    {delimiter}
    Once the user indicates their interest in specific products.
    provide a brief explanation of each product and ask if they have any questions. Essentially, you are delivering a sales pitch.
    you will also solve the user queries about any product from the catalogue in the user message \
    You should keep the user profile in mind while answering the questions.\
    {delimiter}

    {delimiter}
    In the beginning start with a short welcome message and encourage the user to share their requirements.
    {delimiter}
    """

    conversation = [{"role": "system", "content": base_system_message}]

    return conversation

laptop_spec = [
    {
        "name": "laptop_specifications",
        "description": "Laptop specifications captured based on user inputs",
        "parameters": {
            "type": "object",
            "properties": {
                "gpu_intensity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "GPU intensity describes the extent to which a laptop's graphics processing unit is utilized for tasks like gaming, rendering, or computations. GPU intensity size: low, medium, or high.",
                },
                "display_quality": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Display quality of a laptop encompasses resolution, color accuracy, brightness, contrast, refresh rate, and viewing angles, ensuring visual clarity and vibrancy. Display quality: low, medium, or high.",
                },
                "portability": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Portability of a laptop refers to its ease of transport, determined by factors like weight, size, battery life, and durability. Portability: low, medium, or high.",
                },
                "multitasking": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Multitasking refers to a laptop's ability to handle multiple applications or processes simultaneously without significant performance degradation. Multitasking: low, medium, or high.",
                },
                "processing_speed": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Processing speed of a laptop refers to how quickly the CPU executes tasks, influenced by clock speed, core count, and architecture. Processing speed: low, medium, or high.",
                },
                "budget": {
                    "type": "integer",
                    "description": "Budget in the context of a laptop refers to the total amount of money allocated for purchasing a laptop, considering the balance between cost and desired features. Budget value needs to be greater than or equal to 25000 INR.",
                }
            },
            "required": ["gpu_intensity", "display_quality", "portability", "multitasking", "processing_speed", "budget"]
        }
    }
]

def laptop_specifications(gpu_intensity, display_quality, portability, multitasking, processing_speed, budget):

    return f"Based on your inputs, the laptop should have a {gpu_intensity} GPU intensity, {multitasking} multitasking capability, {processing_speed} processing power, {portability} portability and a {display_quality} display, all within the price range of {budget}."

user_requirement_dict = None

# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_chat_completions(input):
    global user_requirement_dict
    MODEL = 'gpt-3.5-turbo'

    response = openai.chat.completions.create(
    model = MODEL,
    messages = input,
    functions = laptop_spec,
    function_call = 'auto',
    seed = 2345)

    function_call = response.choices[0].message.function_call
    print(function_call)
    print(response.choices[0].message.content)

    if function_call is not None:
        function_args = json.loads(function_call.arguments)
        print("function_args", function_args)
        user_requirement_dict = function_args
        print("user_requirement_dict 29 api call", user_requirement_dict)
        # Invoke the laptop_specifications function with extracted arguments
        spec_message = laptop_specifications(
            gpu_intensity=function_args.get("gpu_intensity"),
            display_quality=function_args.get("display_quality"),
            portability=function_args.get("portability"),
            multitasking=function_args.get("multitasking"),
            processing_speed=function_args.get("processing_speed"),
            budget=function_args.get("budget")
        )
        return spec_message
    else:
        return response.choices[0].message.content, user_requirement_dict

product_spec = [
    {
        "name": "product_specifications",
        "description": "specifications of different products(laptops) available at the shop.",
        "parameters": {
            "type": "object",
            "properties": {
                "gpu_intensity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "GPU intensity describes the extent to which a laptop's graphics processing unit is utilized for tasks like gaming, rendering, or computations. GPU intensity size: low, medium, or high.",
                },
                "display_quality": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Display quality of a laptop encompasses resolution, color accuracy, brightness, contrast, refresh rate, and viewing angles, ensuring visual clarity and vibrancy. Display quality: low, medium, or high.",
                },
                "portability": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Portability of a laptop refers to its ease of transport, determined by factors like weight, size, battery life, and durability. Portability: low, medium, or high.",
                },
                "multitasking": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Multitasking refers to a laptop's ability to handle multiple applications or processes simultaneously without significant performance degradation. Multitasking: low, medium, or high.",
                },
                "processing_speed": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Processing speed of a laptop refers to how quickly the CPU executes tasks, influenced by clock speed, core count, and architecture. Processing speed: low, medium, or high.",
                },
                "price": {
                    "type": "integer",
                    "description": "Price of the laptop",
                }
            },
            "required": ["gpu_intensity", "display_quality", "portability", "multitasking", "processing_speed","price"]
        }
    }
]

def product_map_layer(laptop_description):
    delimiter = "#####"

    prompt=f"""
    You are a Laptop Specifications Classifier whose job is to extract the key features of laptops and classify them as per their requirements.
    To analyze each laptop, perform the following steps:
    Step 1: Extract the laptop's primary features from the description {laptop_description}
    Step 2: Call the function 'product_spec' with the extracted features to populate the python dictionary.
    {delimiter}
    Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised:
    {delimiter}
    GPU Intensity:
    - low: <<< if GPU is entry-level such as an integrated graphics processor or entry-level dedicated graphics like Intel UHD >>> , \n
    - medium: <<< if mid-range dedicated graphics like M1, AMD Radeon, Intel Iris >>> , \n
    - high: <<< high-end dedicated graphics like Nvidia RTX >>> , \n

    Display Quality:
    - low: <<< if resolution is below Full HD (e.g., 1366x768). >>> , \n
    - medium: <<< if Full HD resolution (1920x1080) or higher. >>> , \n
    - high: <<< if High-resolution display (e.g., 4K, Retina) with excellent color accuracy and features like HDR support. >>> \n

    Portability:
    - high: <<< if laptop weight is less than 1.51 kg >>> , \n
    - medium: <<< if laptop weight is between 1.51 kg and 2.51 kg >>> , \n
    - low: <<< if laptop weight is greater than 2.51 kg >>> \n

    Multitasking:
    - low: <<< If RAM size is 8 GB, 12 GB >>> , \n
    - medium: <<< if RAM size is 16 GB >>> , \n
    - high: <<< if RAM size is 32 GB, 64 GB >>> \n

    Processing Speed:
    - low: <<< if entry-level processors like Intel Core i3, AMD Ryzen 3 >>> , \n
    - medium: <<< if Mid-range processors like Intel Core i5, AMD Ryzen 5 >>> , \n
    - high: <<< if High-performance processors like Intel Core i7, AMD Ryzen 7 or higher >>> \n
    {delimiter}

    {delimiter}
    Here is an example to categorize:
    "The Dell Inspiron is a versatile laptop that combines powerful performance and affordability. It features an Intel Core i5 processor clocked at 2.4 GHz, ensuring smooth multitasking and efficient computing. With 8GB of RAM and an SSD, it offers quick data access and ample storage capacity. The laptop sports a vibrant 15.6" LCD display with a resolution of 1920x1080, delivering crisp visuals and immersive viewing experience. Weighing just 2.5 kg, it is highly portable, making it ideal for on-the-go usage. Additionally, it boasts an Intel UHD GPU for decent graphical performance and a backlit keyboard for enhanced typing convenience. With a one-year warranty and a battery life of up to 6 hours, the Dell Inspiron is a reliable companion for work or entertainment. All these features are packed at an affordable price of 35,000, making it an excellent choice for budget-conscious users."
     In this laptop, GPU intensity is low , Display quality is medium, Portability is medium, Multitasking is high, Processing speed is medium and price is 35000 INR.
    {delimiter}

    {delimiter}
    If you do not follow these instructions, you will be penalized heavily.
    {delimiter}
    """
    input = f'Follow the above instructions step-by-step and use the function call to complete the task. The laptop desription follows: {laptop_description}.'
    #see that we are using the Completion endpoint and not the Chatcompletion endpoint
    messages=[{"role": "system", "content":prompt },{"role": "user","content":input}]

    response = openai.chat.completions.create(
    model = 'gpt-3.5-turbo',
    messages = messages,
    functions = product_spec,
    function_call = 'auto',
    seed = 2345)

    function_call = response.choices[0].message.function_call

    if function_call is not None:
        function_args = json.loads(function_call.arguments)
        return function_args
    else:
        return response.choices[0].message.content

    return response

# Define a function called moderation_check that takes user_input as a parameter.

def moderation_check(user_input):
    # Call the OpenAI API to perform moderation on the user's input.
    response = openai.moderations.create(input=user_input)

    # Extract the moderation result from the API response.
    moderation_output = response.results[0].flagged
    # Check if the input was flagged by the moderation system.
    if moderation_output:
        # If flagged, return "Flagged"
        return "Flagged"
    else:
        # If not flagged, return "Not Flagged"
        return "Not Flagged"
    
df=pd.read_csv('laptop_data.csv')

df_with_features = df.copy()

df_with_features['laptop_feature'] = df_with_features['Description'].apply(lambda x: product_map_layer(x))


def compare_laptops_with_user(user_req_string):

    # Extracting the budget value from user_requirements
    budget = user_requirement_dict.get('budget')

    # # Creating a copy of the DataFrame and filtering laptops based on the budget
    filtered_laptops = df_with_features.copy()
    filtered_laptops['Price'] = [filtered_laptops['laptop_feature'][i].get('price') for i in range(len(filtered_laptops))]
    filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= budget].copy()
    # filtered_laptops

    # # # Mapping string values 'low', 'medium', 'high' to numerical scores 0, 1, 2
    mappings = {'low': 0, 'medium': 1, 'high': 2}

    # # # Creating a new column 'Score' in the filtered DataFrame and initializing it to 0
    filtered_laptops['Score'] = 0
    # # # Iterating over each laptop in the filtered DataFrame to calculate scores based on user requirements
    for index, row in filtered_laptops.iterrows():
        laptop_values = row['laptop_feature']
        score = 0

    #     # Comparing user requirements with laptop features and updating scores
        for key, user_value in user_requirement_dict.items():
            if key == 'budget':
                continue  # Skipping budget comparison
            laptop_value = laptop_values.get(key, None)
            # print(key, laptop_value)
            laptop_mapping = mappings.get(laptop_value, -1)
            user_mapping = mappings.get(user_value, -1)
            if laptop_mapping >= user_mapping:
                score += 1  # Incrementing score if laptop value meets or exceeds user value

        filtered_laptops.loc[index, 'Score'] = score  # Updating the 'Score' column in the DataFrame

    # Sorting laptops by score in descending order and selecting the top 3 products
    top_laptops = filtered_laptops.drop('laptop_feature', axis=1)
    top_laptops = top_laptops.sort_values('Score', ascending=False).head(3)
    top_laptops_json = top_laptops.to_json(orient='records')  # Converting the top laptops DataFrame to JSON format

    # top_laptops
    return top_laptops_json


def compare_laptops_with_user(user_req_string):

    # Extracting the budget value from user_requirements
    budget = user_requirement_dict.get('budget')

    # # Creating a copy of the DataFrame and filtering laptops based on the budget
    filtered_laptops = df_with_features.copy()
    filtered_laptops['Price'] = [filtered_laptops['laptop_feature'][i].get('price') for i in range(len(filtered_laptops))]
    filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= budget].copy()
    # filtered_laptops

    # # # Mapping string values 'low', 'medium', 'high' to numerical scores 0, 1, 2
    mappings = {'low': 0, 'medium': 1, 'high': 2}

    # # # Creating a new column 'Score' in the filtered DataFrame and initializing it to 0
    filtered_laptops['Score'] = 0
    # # # Iterating over each laptop in the filtered DataFrame to calculate scores based on user requirements
    for index, row in filtered_laptops.iterrows():
        laptop_values = row['laptop_feature']
        score = 0

    #     # Comparing user requirements with laptop features and updating scores
        for key, user_value in user_requirement_dict.items():
            if key == 'budget':
                continue  # Skipping budget comparison
            laptop_value = laptop_values.get(key, None)
            # print(key, laptop_value)
            laptop_mapping = mappings.get(laptop_value, -1)
            user_mapping = mappings.get(user_value, -1)
            if laptop_mapping >= user_mapping:
                score += 1  # Incrementing score if laptop value meets or exceeds user value

        filtered_laptops.loc[index, 'Score'] = score  # Updating the 'Score' column in the DataFrame

    # Sorting laptops by score in descending order and selecting the top 3 products
    top_laptops = filtered_laptops.drop('laptop_feature', axis=1)
    top_laptops = top_laptops.sort_values('Score', ascending=False).head(3)
    top_laptops_json = top_laptops.to_json(orient='records')  # Converting the top laptops DataFrame to JSON format

    # top_laptops
    return top_laptops_json

def recommendation_validation(laptop_recommendation):
    data = json.loads(laptop_recommendation)
    data1 = []
    for i in range(len(data)):
        if data[i]['Score'] > 2:
            data1.append(data[i])

    return data1

def dialogue_mgmt_system():
    global user_requirement_dict
    conversation = initialize_conversation()

    introduction = get_chat_completions(conversation)
    print(introduction, "\n")

    top_3_laptops = None
    user_input = ''
    last_user_requirement_dict = None

    while user_input != "exit":
        global user_requirement_dict

        user_input = input("")

        moderation = moderation_check(user_input)
        if moderation == 'Flagged':
            print("Sorry, this message has been flagged. Please restart your conversation.")
            break

        conversation.append({"role": "user", "content": user_input})

        response_assistant = get_chat_completions(conversation)
        moderation = moderation_check(str(response_assistant))
        if moderation == 'Flagged':
            print("Sorry, this message has been flagged. Please restart your conversation.")
            break
        conversation.append({"role": "assistant", "content": str(response_assistant)})
        print("\n" + str(response_assistant) + "\n")
        print("user_requirement_dict44", user_requirement_dict)

        # Always check if user_requirement_dict is updated
        if user_requirement_dict is not None and user_requirement_dict != last_user_requirement_dict:
            print("Thank you for providing all the information. Kindly wait, while I fetch the products: \n")
            top_3_laptops = compare_laptops_with_user(user_requirement_dict)

            validated_reco = recommendation_validation(top_3_laptops)

            user_message = f"""These are the products that match my requirements. Try to sell these products by explaining each feature : {validated_reco}"""
            conversation.append({"role": "user", "content": user_message})

            recommendation = get_chat_completions(conversation)

            moderation = moderation_check(str(recommendation))
            if moderation == 'Flagged':
                print("Sorry, this message has been flagged. Please restart your conversation.")
                break

            conversation.append({"role": "assistant", "content": str(recommendation)})

            print(str(recommendation) + '\n')

            # Update last_user_requirement_dict
            last_user_requirement_dict = user_requirement_dict

        else:
            # conversation.append({"role": "assistant", "content": str(response_assistant)})
            # print("\n", response_assistant, "\n")

            if user_requirement_dict is not None:
                # If user_requirement_dict is filled but not updated, just continue
                continue
            else:
                # Process normally when user_requirement_dict is None
                pass

if __name__ == '__main__':
    dialogue_mgmt_system()
import openai
import pandas as pd
import time

# Set your OpenAI API key
filepath_key = "Open_API_Key.txt"
with open(filepath_key, "r") as f:
    text = f.read()
openai.api_key = text.strip()

# Load the dataset
input_file = 'laptops.csv'
output_file = 'laptops_description.csv'
df = pd.read_csv(input_file, encoding='ISO-8859-1')  # file contains characters that aren’t UTF-8 encoded.
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces
print(df.columns)  # Print columns to check the exact names
print(df.iloc[0])  # Display the first row to inspect exact column keys

# Prepare the few-shot example for the prompt
few_shot_examples = [
    {
        "role": "user",
        "content": (
            "Brand: Dell\nModel Name: Inspiron\nCore: i5\nCPU Manufacturer: Intel\nClock Speed: 2.4 GHz\n"
            "RAM Size: 8GB\nStorage Type: SSD\nDisplay Type: LCD\nDisplay Size: 15.6\"\nGraphics Processor: Intel UHD\n"
            "Screen Resolution: 1920x1080\nOS: Windows 10\nLaptop Weight: 2.5 kg\nSpecial Features: Backlit Keyboard\n"
            "Warranty: 1 year\nAverage Battery Life: 6 hours\nPrice: 35,000\n"
            "Description: The Dell Inspiron is a versatile laptop that combines powerful performance and affordability. "
            "It features an Intel Core i5 processor clocked at 2.4 GHz, ensuring smooth multitasking and efficient computing. "
            "With 8GB of RAM and an SSD, it offers quick data access and ample storage capacity. The laptop sports a vibrant 15.6\" LCD display "
            "with a resolution of 1920x1080, delivering crisp visuals and an immersive viewing experience. Weighing just 2.5 kg, it is highly portable, "
            "making it ideal for on-the-go usage. Additionally, it boasts an Intel UHD GPU for decent graphical performance and a backlit keyboard "
            "for enhanced typing convenience. With a one-year warranty and a battery life of up to 6 hours, the Dell Inspiron is a reliable companion "
            "for work or entertainment. All these features are packed at an affordable price of 35,000, making it an excellent choice for budget-conscious users."
        )
    },
    {
        "role": "user",
        "content": (
            "Brand: MSI\nModel Name: GL65\nCore: i7\nCPU Manufacturer: Intel\nClock Speed: 2.6 GHz\nRAM Size: 16GB\n"
            "Storage Type: HDD+SSD\nDisplay Type: IPS\nDisplay Size: 15.6\"\nGraphics Processor: NVIDIA GTX\n"
            "Screen Resolution: 1920x1080\nOS: Windows 10\nLaptop Weight: 2.3 kg\nSpecial Features: RGB Keyboard\n"
            "Warranty: 2 years\nAverage Battery Life: 4 hours\nPrice: 55,000\n"
            "Description: The MSI GL65 is a high-performance laptop designed for gaming enthusiasts. Powered by an Intel Core i7 processor running at 2.6 GHz, "
            "it delivers exceptional processing power for smooth gaming and demanding tasks. With 16GB of RAM and a combination of HDD and SSD storage, it offers ample memory "
            "and fast data access. The laptop features a 15.6\" IPS display with a resolution of 1920x1080, ensuring vivid colors and wide viewing angles for an immersive gaming experience. "
            "Equipped with an NVIDIA GTX graphics card, it provides excellent visual performance and smooth gameplay. Weighing just 2.3 kg, it is a portable option for gamers on the move. "
            "The laptop also boasts an RGB keyboard, allowing customizable lighting effects for a personalized gaming setup. With a two-year warranty and a battery life of up to 4 hours, "
            "the MSI GL65 offers reliability and durability. Priced at 55,000, it offers excellent value for money for gamers seeking a powerful gaming laptop."
        )
    }
]

# Function to generate description for each laptop
def generate_description(row):
    prompt = f"""Here are the laptop details:
    Manufacturer: {row['Manufacturer']}
    Model Name: {row['Model Name']}
    Category: {row['Category']}
    Screen Size: {row['Screen Size']}
    Screen: {row['Screen']}
    CPU: {row['CPU']}
    RAM: {row['RAM']}
    Storage: {row['Storage']}
    GPU: {row['GPU']}
    Operating System: {row['Operating System']}
    Operating System Version: {row['Operating System Version']}
    Weight: {row['Weight']}
    Price: {row['Price (Euros)']}
    Description:"""
    
    # Define the conversation structure with the additional instruction
    conversation = [
        {"role": "system", "content": "Use the details provided to generate simple, customer-friendly descriptions for each laptop. The description must be in sentence format and easy to read."
                                      "Note that the few-shot examples may not exactly match the columns in the dataset, so describe based on available information."},
        *few_shot_examples,
        {"role": "user", "content": prompt}
    ]
    
    try:
        # Call the OpenAI API to generate the description
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            max_tokens=150
        )
        # Extract the generated description
        description = response['choices'][0]['message']['content'].strip()
        return description
    except Exception as e:
        print(f"Error generating description for {row['Model Name']}: {e}")
        return "Description not available"

# Apply the function to each row and add the description column
df['Description'] = df.apply(generate_description, axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)
print(f"Descriptions added successfully. Output saved to {output_file}")

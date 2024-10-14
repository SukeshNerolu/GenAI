The data set has the following columns
Columns  = Manufacturer,Model Name,Category,Screen Size,Screen,CPU,RAM, Storage,GPU,Operating System,Operating System Version,Weight,Price (Euros)

The script has few shot prompting that help to develop a description for the laptop product using gpt-3.5-turbo.

Note: openai version lowered to 0.28 for the implemented script. Alternatevily, the following script can be used if latest version is of openai is being used.



try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Update to "text-davinci-003" if turbo is unsupported
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        description = response.choices[0].text.strip()
        return description
    except Exception as e:
        print(f"Error generating description for {row['Model Name']}: {e}")
        return "Description not available"

# Apply the function to each row and add the description column
df['Description'] = df.apply(generate_description, axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)
print(f"Descriptions added successfully. Output saved to {output_file}")
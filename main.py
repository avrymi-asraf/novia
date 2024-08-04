import json
import PyPDF2
import json
from openai import OpenAI

SYSTEM_PROMPT = """You analyze the suitability of employees for the position, be strict and answer according to the requirements. Make sure the format matches json"""


def read_config():
    with open("config.json") as f:
        data = json.load(f)
    return data


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def analyze_cv(pdf_path, api_key):
    client = OpenAI(api_key=api_key)
    cv_text = extract_text_from_pdf(pdf_path)
    prompt = f"""
        Extract the following information from the given CV text:
        1. Candidate's name
        2. Key skills (list of 5-10 most relevant skills)
        3. Years of experience
        4. Education level
        5. Most recent job title and company
        6. A brief summary of the candidate's profile (2-3 sentences)

        CV Text:
        {cv_text}

        Please provide the information in a JSON format with the following keys:
        "name", "skills", "years_of_experience", "education_level", "recent_job", "summary"
        """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
    )
    extracted_info = None
    count = 4
    while count > 0:
        try:
            extracted_info = json.loads(response.choices[0].message.content)
            break
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON response from API analyze_cv")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            count -= 1

    return extracted_info


def evaluate_candidate_fit(job_description, cv_info, api_key):
    client = OpenAI(api_key=api_key)

    prompt = f"""
    Given the following job description and candidate information, evaluate the candidate's fit for the role:

    Job Description:
    {job_description}

    Candidate Information:
    Name: {cv_info['name']}
    Skills: {', '.join(cv_info['skills'])}
    Years of Experience: {cv_info['years_of_experience']}
    Education Level: {cv_info['education_level']}
    Most Recent Job: {cv_info['recent_job']}
    Summary: {cv_info['summary']}

    Please provide:
    1. A brief summary (2-3 sentences) of the candidate's match to the role.
    2. A classification of the candidate's fit into one of three categories:
       A: Good fit
       B: Medium fit
       C: Not a good fit

    Return your evaluation in JSON format with the following keys:
    "summary", "classification"
    """

    evaluation = None
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
    )
    count = 4
    while count > 0:
        try:
            evaluation = json.loads(response.choices[0].message.content)
            break
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON response from API evaluate")
            # print(response.choices[0].message.content)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            count -= 1

    return evaluation


# Example usage
if __name__ == "__main__":
    config = read_config()
    pdfs_path = config["pdf_path"]
    api_key = config["openai"]["api_key"]
    job_description = config["job_description"]
    for pdf_path in pdfs_path:
        cv_info = analyze_cv(pdf_path, api_key)

        if cv_info:
            evaluation = evaluate_candidate_fit(job_description, cv_info, api_key)
            if evaluation:
                print(f"Evaluation for {cv_info['name']}:")
                print(json.dumps(evaluation, indent=2))

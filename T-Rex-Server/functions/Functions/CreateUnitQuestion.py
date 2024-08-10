import functions_framework
import openai
from openai._client import OpenAI
from random import randint
from flask import jsonify
import json
from . import FireBaseFunctions as FF

client = OpenAI(api_key = "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")

MODEL = "gpt-3.5-turbo-1106"

GENERAL_QUESTION_PROMPT = """Based on this course plan:
course_curriculum

Write question_type for the unit "unit_title" that can be answered by the unit content.
The unit content: unit_content.

Output your answer as a json built like this:
answer_json_stracture
"""

SINGLE_CHOICE_PROMPT = (
    "a single choice question with n_possible_answers possible answers",
    """{
    'Question': '', #the question
    'Answer': '', #the correct answer
    'IncorrectAnswers': [''], #a list of incorrect answers
    'Explanation': '' #explaining why the correct answer is correct
}"""
)

MULTI_CHOICE_PROMPT = (
    "a multi choice question with n_possible_answers possible answers and n_correct_answers correct answers",
    """{
    'Question': '', #the question
    'Answers': [''], #a list of correct answers
    'IncorrectAnswers': [''], #a list of incorrect answers
    'Explanation': '' #explaining why the correct answers are correct
}"""
)

OPEN_QUESTION_PROMPT = (
    "an open question",
    """{
    'Question': '', #the question
    'Answer': '', #the correct answer
}"""
)

FILL_THE_BLANK_PROMPT = (
    "a cloze test type question with n_blanks blanks and n_possible_answers possible answers for each blank",
    """{
    'Question': '', # the question, with blank words represented as _____ (5 underscores), for example: "The _____ is the study of the mind and behavior."
    'Answers': [ # a list of dictionaries, each dictionary represents a blank and its possible answers 
        { # a dictionry representing the possible answers for the first blank
            "Answer": '', # the correct answer
            "IncorrectAnswers": [''] # a list of incorrect answers
        } 
    ], 
    'Explanation': '' #explaining why the correct answer is correct
}"""
)


@functions_framework.http
def CreateUnitQuestion(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    try:
        session_id = request_json['SessionID']
        user_id = request_json['UserID']
        FF.auth.verify_id_token(session_id)
    except:
        return (jsonify({'Message': 'User not verified', "Code":1}), 200)

    relevant_keys = [ "unit_id", "unit_title", "unit_content", "q_type", "course_curriculum"]
    for key in relevant_keys:
        if key not in request_json or len(request_json[key])==0:
            return (jsonify({"Message":"Missing some essential parameters", "Code":1}), 200)
    if 'former_questions' not in request_json:
        return (jsonify({"Message":"Missing some essential parameters", "Code":1}), 200)
    
    owner_id = None
    try:
        owner_id = FF.GetDocumentField("Units", request_json['unit_id'], "OwnerID")
    except:
        return (jsonify({"Message":"Error when retreiving data", "Code":1}), 200)
    if owner_id is None or user_id!=owner_id:
        return (jsonify({"Message":"User is not the owner of the unit", "Code":1}), 200)
    
    base_question_prompt = GENERAL_QUESTION_PROMPT.replace('unit_title', request_json['unit_title'])
    base_question_prompt = base_question_prompt.replace('course_curriculum', request_json['course_curriculum'])
    base_question_prompt = base_question_prompt.replace('unit_content', request_json['unit_content'])
    
    question_type = ""
    answer_json_stracture = ""
    match request_json['q_type']:
        case 'Single Choice':
            question_type, answer_json_stracture = SINGLE_CHOICE_PROMPT
            n_possible_answers = randint(3,6)
            question_type = question_type.replace('n_possible_answers', str(n_possible_answers))
        case 'Multi Choice':
            question_type, answer_json_stracture = MULTI_CHOICE_PROMPT
            n_possible_answers = randint(3,6)
            question_type = question_type.replace('n_possible_answers', str(n_possible_answers))
            n_correct_answers = randint(2,n_possible_answers)
            question_type = question_type.replace('n_correct_answers', str(n_correct_answers))
        case 'Open Question':
            question_type, answer_json_stracture = OPEN_QUESTION_PROMPT
        case 'Fill The Blank':
            question_type, answer_json_stracture = FILL_THE_BLANK_PROMPT
            n_blanks = randint(2,4)
            question_type = question_type.replace('n_blanks', str(n_blanks))
            n_possible_answers = randint(1,4)
            question_type = question_type.replace('n_possible_answers', str(n_possible_answers))
        case _:
            return (jsonify({"Message":"Invalid question type", "Code":1}), 200)

    current_q_prompt = base_question_prompt.replace('question_type', question_type)
    current_q_prompt = current_q_prompt.replace('answer_json_stracture', answer_json_stracture)

    if len(request_json['former_questions'])>0:
        old__q = "\n\nHere is a list of the questions you already wrote for this unit (make sure not to repeat them):\n"
        for q in request_json['former_questions']:
            old__q += f"{q}\n"
        current_q_prompt += old__q
    
    try:
        owner_id = FF.GetDocumentField("Units", request_json['unit_id'], "OwnerID")
        if owner_id is None or user_id!=owner_id:
            return (jsonify({"Message":"User is not the owner of the unit", "Code":1}), 200)
    
        response = client.chat.completions.create(
            model=MODEL,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to write textbook exercise for students."},
                {"role": "user", "content": current_q_prompt}
            ]
        )
        question = json.loads(response.choices[0].message.content)
        final_question = FF.SaveQuestion(user_id, request_json['q_type'], request_json['unit_id'], question)
        return (jsonify({'Question':final_question, "Code":0}), 200)
    except:
        return (jsonify({"Message":"OpenAI API Error", "Code":1}), 200)
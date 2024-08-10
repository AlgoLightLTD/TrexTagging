import functions_framework
from flask import jsonify
from . import FireBaseFunctions as FF

@functions_framework.http
def DeleteCourse(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    try:
        session_id = request_json['SessionID']
        user_id = request_json['UserID']
        FF.auth.verify_id_token(session_id)
    except:
        return (jsonify({'Message': 'User not verified', "Code":1}), 200)

    relevant_keys = [ "course_id"]
    for key in relevant_keys:
        if key not in request_json or len(request_json[key])==0:
            return (jsonify({"Message":"Missing some essential parameters", "Code":1}), 200)
    
    try:
        owner_id = FF.GetDocumentField("Courses", request_json['course_id'], "OwnerID")
        if owner_id is None or user_id!=owner_id:
            return (jsonify({"Message":"User is not the owner of the unit", "Code":1}), 200)
    
        # delete all the sub parts of a course
        parts_ids = FF.GetDocumentField("Courses", request_json['course_id'], "PartsIDs")
        for part_id in parts_ids:
            chapters_ids = FF.GetDocumentField("Parts", part_id, "ChaptersIDs")
            for chapter_id in chapters_ids:
                units_ids = FF.GetDocumentField("Chapters", chapter_id, "UnitsIDs")
                for unit_id in units_ids:
                    questions_ids = FF.GetDocumentField("Units", unit_id, "QuestionsIDs")
                    for question_id in questions_ids:
                        FF.DeleteDocument("Questions", question_id)
                    FF.DeleteDocument("Units", unit_id)
                FF.DeleteDocument("Chapters", chapter_id)
            FF.DeleteDocument("Parts", part_id)
        FF.DeleteDocument("Courses", request_json['course_id'])

        users_data_id = FF.GetDocumentsByFilter("UsersData", "UserID", user_id)[0]['ID']
        FF.UpdateDocument("UsersData", users_data_id, {"CoursesIDs":FF.firestore.ArrayRemove([request_json['course_id']])})

        return (jsonify({"Code":0}), 200)
    except:
        return (jsonify({"Message":"Something went wrong", "Code":1}), 200)
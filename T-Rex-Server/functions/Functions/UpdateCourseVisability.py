import functions_framework
from flask import jsonify
from . import FireBaseFunctions as FF

@functions_framework.http
def UpdateCourseVisability(request):
    request_json = request.get_json(silent=True)
    request_args = request.args
    
    try:
        session_id = request_json['SessionID']
        user_id = request_json['UserID']
        FF.auth.verify_id_token(session_id)
    except:
        return (jsonify({'Message': 'User not verified', "Code":1}), 200)

    relevant_keys = [ "course_id" ]
    for key in relevant_keys:
        if key not in request_json or len(request_json[key])==0:
            return (jsonify({"Message":"Missing some essential parameters", "Code":1}), 200)
    if 'public' not in request_json or request_json['public'] is None:
            return (jsonify({"Message":"Missing some essential parameters", "Code":1}), 200)
    
    try:
        owner_id = None
        owner_id = FF.GetDocumentField("Courses", request_json['course_id'], "OwnerID")
        if owner_id is None or user_id!=owner_id:
            return (jsonify({"Message":"User is not the owner of the unit", "Code":1}), 200)
        
        FF.UpdateDocument("Courses", request_json['course_id'], {"Public":request_json['public']})
        return (jsonify({"Message":"Unit content updated", "Code":0}), 200)
    except:
        return (jsonify({"Message":"Something went wrong", "Code":1}), 200)
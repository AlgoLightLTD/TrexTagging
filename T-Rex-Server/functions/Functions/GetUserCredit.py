import functions_framework
from flask import jsonify
from . import FireBaseFunctions as FF


@functions_framework.http
def GetUserCredit(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    try:
        session_id = request_json['SessionID']
        user_id = request_json['UserID']
        FF.auth.verify_id_token(session_id)
    except:
        return (jsonify({'Message': 'User not verified', "Code":1}), 200)
    
    try:
        user_data = FF.GetDocumentsByFilter("UsersData", "UserID", user_id)
        if len(user_data) == 0:
            user_data_id = FF.GetNewEmptyDocumentID("UsersData")
            user_data = {
                "Credit": 0.0,
                "CoursesIDs": [],
                "UserID": user_id,
                "ID": user_data_id
            }
            FF.db.collection("UsersData").document(user_data_id).set(user_data)
        else:
            user_data = user_data[0]
        return (jsonify({"Credit":user_data["Credit"], "Code":0}), 200)
    except:
        return (jsonify({"Message":"Something went wrong", "Code":1}), 200)
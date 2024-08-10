import firebase_admin
from firebase_admin import firestore
from firebase_admin import auth

# initialize firebase
cred = firebase_admin.credentials.Certificate(
    {
        "type": "service_account",
        "project_id": "intellicourse",
        "private_key_id": "44bba6ecae475d3d674567e160f533c0befddb5c",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC4Y85h2hQIt8A0\nzB61I7FhwTssp7s2HOzhpDcf0wW51mEc9SDe2pSb3WdxIskecwopyYpHphKjkeHj\nHuL0+CO36OAl2Pi7w27CmE/FO3/67jFP1mzkdHYs5UpuSX+O595Mw5MjCbJvLTox\ncxp3CPol3sceqnyc6ysskQCCgWJhTnD2xDe9s+eZDZhla1qeH/EeG/2quTR6VWA6\nibSTxtKfUtT4pApjHyfbGbgJwqj6dZpbHxym4TtBgz1RLfzOjrb2TVZI6rBPIJVH\nOTZGl/ze7Nk9bh3idJzLZ5ks/6pOFqlU4h+i+k2Ym3K4ncCIifS4CFO1WRSjuTBB\n6EucUHXJAgMBAAECggEACBPAgH3T6daEHMFM87oMClBeHg8nlBpr8AMdsmRajbD2\nkRUAgHKfwZIkdrVuN59qHw7SjV/uKhcjsityCAbEqHOS5LremSRZpObqJlAnGOAN\nQ0Y31UFwGc2zgdBGE7BUMcBRdjP1tTQMfApi6uWhm5No3TmHpZIRp+1JfJixWA91\n5cr9f+guUiDtiKARwtCvEuXF6FH0RezIyrKBJPyEEot15VJInd+Rqz3WFXfRq7XX\nWya7pxvF9W1YCe/3xmWFOlrz3O9k2ms068tklXBr4QYvPmVg7Q/YkF6iUA5SKjv/\nCCG7vXL0iLcmrcmLdnMXYOcQqBW8R2r5G+Dxi7ipuQKBgQD3l1b4TI7S3ydDzacC\nQBzG4bUmA04bFgWRwRTG4/xjdut0pBT8AuybD4+05tcU6UkbsfCB2TA+JfW3Xjn0\n/LzWVFhXPwXGH6PKo7zdvYfqHGVzaoe8nZ7YVvf+O9YMGWNe1OiaNbWkG7DY3T+G\n7MSjFuneU5wSQDop6d2NWVcT9QKBgQC+pvfXOR0I/Lo0h8+L540j0rdLJ3AoqoAa\nnV3vnpxXbrarxmbJC3tPp4NLl4HhqNsDxESLNl8yS9AX/SXnW6O/zGbo53NbezRh\n+r4yJYurAyTLHem5lbwp4mmwT8rpZztnf9UIAQZcj6+ExebvKiUiqmXhbk5ygnnh\nTk13S4+KBQKBgQDouK79yvBAk5KTr904R00FP6iG53pmCUgI3aUK4ccdMu1UgNpc\nmt8NTQ416vjl5fVa++FmG+C1ufaBeicd+OYCy9sRfuKNfLX3RfrDrl/vlOtjFhIq\nHznsfaqypEmoq8B0pclREgT/ESwkW6pXx3t5FJOU75/2o7pIJVOjhgNMdQKBgBID\nVIdRLh0bQ7yS2Fkvl3Y7AnZnU4GEGnZpR+bL5ZmdLEUuiaeVHiF4gj8yIWNgwNCq\nSo+co4HtB8w8bshdMCt+9Hos5AlQqrJXLoEuavPNaDyIpvyR2SVb7wBpiZJj3oj4\n2vfWoPM94Cd7S0ZthPSxhM63zVGkYj+XVlpfGUDBAoGBAKnv3vz197JzcUagGY+X\nkBGG0+xZiKRHWtQ9PC61kjaD51tfaQowhVMz0Q9fUDOqi2nCJx4r+4H8oOchlPQd\n+QBOrqdgXHkrE/f/gVrvZb5nj36snKClUgGSBdOCZDLvNajDY7sS1pavRm8e+21J\n0aSohk+PdB01+NqK3DMirusM\n-----END PRIVATE KEY-----\n",
        "client_email": "firebase-adminsdk-g87af@intellicourse.iam.gserviceaccount.com",
        "client_id": "109841619335333242853",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-g87af%40intellicourse.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }
)
firebase_admin.initialize_app(cred)
db = firestore.client()

def GetNewEmptyDocumentID(collection_name:str):
    return db.collection(collection_name).document().id

def GetDocument(collection_name:str, document_id: str):
    doc_ref = db.collection(collection_name).document(document_id)
    doc = doc_ref.get()
    if doc.exists:  # Check if the document exists
        return doc
    else:
        return None

def GetDocumentField(collection_name:str, document_id: str, field: str):
    doc = GetDocument(collection_name, document_id)
    if doc is not None:  # Check if the document exists
        return doc.to_dict()[field]
    else:
        return None

def UpdateDocument(collection_name:str, document_id: str, data: dict):
    doc_ref = db.collection(collection_name).document(document_id)
    doc_ref.update(data)

def GetDocumentsByFilter(collection_name: str, field: str, value: str):
    final_documents = []
    documents = db.collection(collection_name).where(field_path=field, op_string='==', value=value).stream()
    for document in documents:
        final_documents.append(document.to_dict())

    return final_documents

def DeleteDocument(collection_name: str, document_id: str):
    db.collection(collection_name).document(document_id).delete()
import os
import json
from plaid import Client
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser

client = Client(
    client_id=os.environ['PLAID_CLIENT_ID'],
    secret=os.environ['PLAID_SECRET'],
    environment='sandbox'
)

def lambda_handler(event, context):
    try:
        user_id = event["queryStringParameters"]["userId"]
        request = LinkTokenCreateRequest(
            user=LinkTokenCreateRequestUser(client_user_id=user_id),
            client_name="Chimera SubSaver",
            products=["transactions"],
            country_codes=["US"],
            language="en",
        )
        response = client.link_token_create(request)
        return {
            "statusCode": 200,
            "body": json.dumps({"link_token": response["link_token"]})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

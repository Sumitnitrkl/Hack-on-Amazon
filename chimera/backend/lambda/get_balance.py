import os
import json
import boto3
from plaid import Client
from plaid.model.accounts_balance_get_request import AccountsBalanceGetRequest

client = Client(
    client_id=os.environ["PLAID_CLIENT_ID"],
    secret=os.environ["PLAID_SECRET"],
    environment="sandbox"
)

dynamodb = boto3.resource("dynamodb")
users_table = dynamodb.Table(os.environ["USERS_TABLE"])

def lambda_handler(event, context):
    try:
        user_id = event["queryStringParameters"]["userId"]
        user_data = users_table.get_item(Key={"userId": user_id})

        if "Item" not in user_data:
            raise Exception("User not found")

        access_token = user_data["Item"]["plaidAccessToken"]

        request = AccountsBalanceGetRequest(access_token=access_token)
        response = client.accounts_balance_get(request)

        return {
            "statusCode": 200,
            "body": json.dumps(response["accounts"])
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

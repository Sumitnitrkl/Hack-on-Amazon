import os
import json
import boto3
from plaid import Client
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from datetime import datetime, timedelta

client = Client(
    client_id=os.environ['PLAID_CLIENT_ID'],
    secret=os.environ['PLAID_SECRET'],
    environment='sandbox'
)

dynamodb = boto3.resource('dynamodb')
users_table = dynamodb.Table(os.environ['USERS_TABLE'])


def lambda_handler(event, context):
    try:
        user_id = event["queryStringParameters"]["userId"]
        user_data = users_table.get_item(Key={"userId": user_id})

        if "Item" not in user_data:
            raise Exception("User not found")

        access_token = user_data["Item"]["plaidAccessToken"]
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        request = TransactionsGetRequest(
            access_token=access_token,
            start_date=start_date,
            end_date=end_date,
            options=TransactionsGetRequestOptions(count=100)
        )

        response = client.transactions_get(request)
        transactions = response["transactions"]

        return {
            "statusCode": 200,
            "body": json.dumps({"transactions": transactions})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
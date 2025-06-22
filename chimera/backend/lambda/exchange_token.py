import os
import json
import boto3
from plaid import Client
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest

client = Client(
    client_id=os.environ['PLAID_CLIENT_ID'],
    secret=os.environ['PLAID_SECRET'],
    environment='sandbox'
)

dynamodb = boto3.resource('dynamodb')
users_table = dynamodb.Table(os.environ['USERS_TABLE'])

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        public_token = body['public_token']
        user_id = body['user_id']

        exchange_request = ItemPublicTokenExchangeRequest(public_token=public_token)
        exchange_response = client.item_public_token_exchange(exchange_request)

        access_token = exchange_response['access_token']
        item_id = exchange_response['item_id']

        users_table.put_item(Item={
            'userId': user_id,
            'plaidAccessToken': access_token,
            'plaidItemId': item_id
        })

        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'success'})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
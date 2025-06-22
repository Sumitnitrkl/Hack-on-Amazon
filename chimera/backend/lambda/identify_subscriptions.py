import os
import json
import boto3
import boto3.session
import uuid

bedrock = boto3.client(service_name="bedrock-runtime")
dynamodb = boto3.resource("dynamodb")
subs_table = dynamodb.Table(os.environ["SUBSCRIPTIONS_TABLE"])


def lambda_handler(event, context):
    try:
        user_id = event["queryStringParameters"]["userId"]
        transactions = json.loads(event["body"])  # assume frontend passes filtered list

        prompt = f"""
        Analyze these transactions and find recurring subscription patterns. For each, return:
        - Service Name
        - Recurrence (monthly/yearly)
        - Amount (USD)
        - Description

        Transactions: {transactions}
        Return result as a JSON list.
        """

        response = bedrock.invoke_model(
            modelId=os.environ["BEDROCK_MODEL_ID"],
            body=json.dumps({"prompt": prompt, "max_tokens": 300, "temperature": 0.3}),
            contentType="application/json",
            accept="application/json"
        )

        results = json.loads(response["body"].read().decode())

        for sub in results:
            subs_table.put_item(Item={
                "userId": user_id,
                "subscriptionId": str(uuid.uuid4()),
                "serviceName": sub["Service Name"],
                "amount": sub["Amount"],
                "description": sub["Description"],
                "recurrence": sub["Recurrence"],
                "status": "pending_review"
            })

        return {
            "statusCode": 200,
            "body": json.dumps({"identified": results})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
import os
import json
import boto3
import uuid

bedrock = boto3.client("bedrock-runtime")
dynamodb = boto3.resource("dynamodb")
users_table = dynamodb.Table(os.environ["USERS_TABLE"])
savings_table = dynamodb.Table(os.environ["SAVINGS_TABLE"])

def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        user_id = body["user_id"]
        balance = float(body["balance"])  # from Plaid
        buffer = float(body["buffer"])    # from DB or frontend
        goal = body["goal"]

        excess = balance - buffer
        if excess <= 0:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No excess to sweep."})
            }

        prompt = f"""
        The user has ${excess:.2f} more than their buffer of ${buffer:.2f}.
        Their saving goal is: {goal}.
        Suggest that they move this money to savings and show potential interest (assume 4.5% APY).
        """

        response = bedrock.invoke_model(
            modelId=os.environ["BEDROCK_MODEL_ID"],
            body=json.dumps({"prompt": prompt, "max_tokens": 200, "temperature": 0.5}),
            contentType="application/json",
            accept="application/json"
        )

        suggestion_text = json.loads(response["body"].read().decode())["message"]

        savings_table.put_item(Item={
            "userId": user_id,
            "suggestionId": str(uuid.uuid4()),
            "suggestedAmount": excess,
            "goal": goal,
            "text": suggestion_text,
            "status": "pending_approval"
        })

        return {
            "statusCode": 200,
            "body": json.dumps({"sweep": suggestion_text})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

import os
import json
import boto3

bedrock = boto3.client("bedrock-runtime")
dynamodb = boto3.resource("dynamodb")
subs_table = dynamodb.Table(os.environ["SUBSCRIPTIONS_TABLE"])

def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        user_id = body["user_id"]
        subscription_id = body["subscription_id"]
        service = body["service"]
        amount = body["amount"]

        prompt = f"""
        The user is subscribed to {service} for ${amount}/month.
        - Suggest a cheaper plan if available (e.g., student/family).
        - Draft a polite cancellation or downgrade email.
        - Try to include a direct account management URL if known.
        Format result as:
        {{
            "tip": "...",
            "email": "...",
            "url": "..."
        }}
        """

        response = bedrock.invoke_model(
            modelId=os.environ["BEDROCK_MODEL_ID"],
            body=json.dumps({"prompt": prompt, "max_tokens": 400, "temperature": 0.4}),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read().decode())

        subs_table.update_item(
            Key={"userId": user_id, "subscriptionId": subscription_id},
            UpdateExpression="SET aiSuggestionText = :tip, actionUrl = :url, aiEmailText = :email, #s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":tip": result["tip"],
                ":url": result["url"],
                ":email": result["email"],
                ":s": "action_suggested"
            }
        )

        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

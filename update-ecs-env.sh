#!/bin/bash

# Update ECS task definition with API_TOKEN environment variable
# Usage: ./update-ecs-env.sh "your-api-token-here"

if [ -z "$1" ]; then
    echo "Usage: $0 <api-token>"
    echo "Example: $0 'abc123-your-secure-token'"
    exit 1
fi

API_TOKEN="$1"
TASK_DEFINITION_NAME="influencer-brain-task"
REGION="us-east-1"

echo "Getting current task definition..."

# Get the current task definition
TASK_DEF=$(aws ecs describe-task-definition \
    --task-definition $TASK_DEFINITION_NAME \
    --region $REGION \
    --query 'taskDefinition')

# Create a new task definition JSON with the API_TOKEN added
echo "$TASK_DEF" | jq --arg token "$API_TOKEN" '
    # Remove fields that AWS will regenerate
    del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .placementConstraints, .compatibilities, .registeredAt, .registeredBy, .enableFaultInjection) |
    # Add API_TOKEN to environment variables
    .containerDefinitions[0].environment += [{"name": "API_TOKEN", "value": $token}]
' > new-task-def.json

echo "Registering new task definition with API_TOKEN..."

# Register the new task definition
NEW_TASK_DEF_ARN=$(aws ecs register-task-definition \
    --region $REGION \
    --cli-input-json file://new-task-def.json \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

echo "New task definition registered: $NEW_TASK_DEF_ARN"

# Update the service to use the new task definition
echo "Updating ECS service..."
aws ecs update-service \
    --cluster influencer-brain \
    --service influencer-brain-task-service-wepvg641 \
    --task-definition $NEW_TASK_DEF_ARN \
    --region $REGION

# Clean up
rm new-task-def.json

echo "✅ ECS service updated with API_TOKEN environment variable!"
echo "Your deployment script will now work with authentication."
